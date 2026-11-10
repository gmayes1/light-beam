import CoreImage
import Vision
import Foundation

struct Look: Codable {
    var bird_ev = 0.15, bird_warm_k = 150.0, shadow_lift = 0.25, eye_ev = 0.2, bird_sharp = 0.35, bird_vibrance = 0.2
    var bird_contrast = 1.0, bird_highlights = 1.0
    var env_ev_far = -0.55, env_cool_k = 250.0, falloff = 1.2, shadow_cool_k = 0.0, env_highlight_warm_k = 0.0
    var burn = 0.25
    var global_contrast = 1.06, global_vibrance = 0.08
    var wb_strength = 0.5
    var bg_denoise = 1.0
    var hot_colour = 0.6
}
struct Recipe: Decodable {
    var raw: String; var out: String; var crop: [Double]?; var out_width: Double?
    var look: [String: Double]?; var auto_crop: Bool?; var ratio: String?; var reference: String?
    var destination: String?; var min_long_px: Double?; var max_effective_iso: Double?; var sharp_snr: Double?
    var stack: [String]?; var stack_auto: Bool?; var stack_min_iso: Double?
}
if CommandLine.arguments.count < 2 { FileHandle.standardError.write("usage: look_pipeline recipe.json\n".data(using: .utf8)!); exit(2) }
let recipe = try JSONDecoder().decode(Recipe.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
var report: [String: Any] = ["raw": recipe.raw]
var warnings: [String] = []

let ctx = CIContext(options: [.cacheIntermediates: false, .memoryTarget: 4096])
let linear = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!

func readback(_ img: CIImage, _ r: CGRect, scale s: CGFloat, gamma: Bool = true) -> (px: [Float], w: Int, h: Int) {
    let scaled = img.cropped(to: r).transformed(by: CGAffineTransform(translationX: -r.minX, y: -r.minY).concatenating(CGAffineTransform(scaleX: s, y: s)))
    let w = max(1, Int(r.width * s)), h = max(1, Int(r.height * s))
    var px = [Float](repeating: 0, count: w * h * 4)
    ctx.render(gamma ? scaled.applyingFilter("CILinearToSRGBToneCurve") : scaled, toBitmap: &px, rowBytes: w * 16,
               bounds: CGRect(x: 0, y: 0, width: w, height: h), format: .RGBAf, colorSpace: nil)
    return (px, w, h)
}
func lum(_ p: [Float], _ i: Int) -> Float { 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2] }
func percentile(_ v: [Float], _ q: Double) -> Float { guard !v.isEmpty else { return 0 }; let s = v.sorted(); return s[min(s.count - 1, Int(Double(s.count - 1) * q))] }
func smooth(_ e0: Float, _ e1: Float, _ x: Float) -> Float { let t = max(0, min(1, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t) }
func temperature(_ img: CIImage, warmerBy k: Double) -> CIImage {
    k == 0 ? img : img.applyingFilter("CITemperatureAndTint", parameters: ["inputNeutral": CIVector(x: 6500, y: 0), "inputTargetNeutral": CIVector(x: 6500 - k, y: 0)])
}

func grey(_ img: CIImage) -> CIImage {
    img.applyingFilter("CIColorMatrix", parameters: ["inputRVector": CIVector(x: 1, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 1, y: 0, z: 0, w: 0),
                                                     "inputBVector": CIVector(x: 1, y: 0, z: 0, w: 0), "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)])
}

func lumaKey(_ img: CIImage, lo: Double, hi: Double) -> CIImage {
    let mid = (lo + hi) / 2
    return img.applyingFilter("CILinearToSRGBToneCurve").applyingFilter("CIColorControls", parameters: ["inputSaturation": 0])
        .applyingFilter("CIToneCurve", parameters: ["inputPoint0": CIVector(x: 0, y: 0), "inputPoint1": CIVector(x: lo, y: 0),
            "inputPoint2": CIVector(x: mid, y: 0.5), "inputPoint3": CIVector(x: hi, y: 1), "inputPoint4": CIVector(x: 1, y: 1)])
}
func mult(_ a: CIImage, _ b: CIImage) -> CIImage { a.applyingFilter("CIMultiplyCompositing", parameters: ["inputBackgroundImage": b]) }

let hotKernel = try CIKernel.kernels(withMetalString: """
#include <CoreImage/CoreImage.h>
extern "C" { namespace coreimage {
[[ stitchable ]] float4 hotColour(sample_t s, float amount) {
    float3 c = s.rgb;
    float mx = max(c.r, max(c.g, c.b)), mn = min(c.r, min(c.g, c.b));
    float sat = mx > 1e-4 ? (mx - mn) / mx : 0.0;
    float t = smoothstep(0.86, 1.0, mx) * smoothstep(0.35, 0.8, sat) * amount;
    float y = dot(c, float3(0.2126, 0.7152, 0.0722));
    c = mix(c, float3(y), t * 0.5);
    float3 k = 0.88 + 0.12 * (1.0 - exp(-(c - 0.88) / 0.12));
    c = select(c, k, c > 0.88);
    return float4(c, s.a);
}
}}
""").first as! CIColorKernel

let raw = CIRAWFilter(imageURL: URL(fileURLWithPath: recipe.raw))!
if raw.supportedDecoderVersions.contains(.version9) { raw.decoderVersion = .version9 }
if raw.isHighlightRecoverySupported { raw.isHighlightRecoveryEnabled = true }
let full = raw.outputImage!.extent
let rb = Int(full.width) * 8
func exif(_ tags: [String]) -> [String: Double] {
    let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    p.arguments = ["exiftool", "-j", "-n"] + tags.map { "-" + $0 } + [recipe.raw]
    let pipe = Pipe(); p.standardOutput = pipe; try? p.run()
    let d = pipe.fileHandleForReading.readDataToEndOfFile(); p.waitUntilExit()
    guard let row = ((try? JSONSerialization.jsonObject(with: d)) as? [[String: Any]])?.first else { return [:] }
    var out: [String: Double] = [:]; for t in tags { if let n = row[t] as? NSNumber { out[t] = n.doubleValue } }; return out
}
func displayAF(_ x: Double, _ y: Double, _ w: Double, _ h: Double, _ iw: Double, _ ih: Double, orientation o: Int, in e: CGRect) -> CGRect {
    var cx = x, cy = y, bw = w, bh = h, sw = iw, sh = ih
    switch o {
    case 3: cx = iw - x; cy = ih - y
    case 6: cx = ih - y; cy = x; bw = h; bh = w; sw = ih; sh = iw
    case 8: cx = y; cy = iw - x; bw = h; bh = w; sw = ih; sh = iw
    default: break
    }
    let sx = Double(e.width) / sw, sy = Double(e.height) / sh
    return CGRect(x: Double(e.minX) + (cx - bw / 2) * sx, y: Double(e.maxY) - (cy + bh / 2) * sy, width: bw * sx, height: bh * sy)
}
let meta = exif(["AFAreaXPosition", "AFAreaYPosition", "AFAreaWidth", "AFAreaHeight", "AFImageWidth", "AFImageHeight", "ISO", "Orientation"])
let iso = meta["ISO"] ?? 0
var afFull: CGRect? = nil
if let x = meta["AFAreaXPosition"], let y = meta["AFAreaYPosition"], let w = meta["AFAreaWidth"], let h = meta["AFAreaHeight"],
   let iw = meta["AFImageWidth"], let ih = meta["AFImageHeight"], w > 0 {
    afFull = displayAF(x, y, w, h, iw, ih, orientation: Int(meta["Orientation"] ?? 1), in: full)
}
report["iso"] = iso

let binDir = ProcessInfo.processInfo.environment["LIGHT_BEAM_BIN"] ?? (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])).resolvingSymlinksInPath().deletingLastPathComponent().path
func run(_ exe: String, _ args: [String], env extra: [String: String] = [:]) -> String {
    let p = Process(); p.executableURL = URL(fileURLWithPath: exe); p.arguments = args
    p.environment = ProcessInfo.processInfo.environment.merging(extra) { $1 }
    let pipe = Pipe(); p.standardOutput = pipe; p.standardError = FileHandle.nullDevice
    try? p.run(); let d = pipe.fileHandleForReading.readDataToEndOfFile(); p.waitUntilExit()
    return String(decoding: d, as: UTF8.self)
}
func burstMates() -> [String] {
    let dir = (recipe.raw as NSString).deletingLastPathComponent
    let name = (recipe.raw as NSString).lastPathComponent
    guard let num = Int(name.filter(\.isNumber)) else { return [] }
    let near = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? [])
        .filter { $0.uppercased().hasSuffix(".NEF") && !$0.hasPrefix("._") && abs((Int($0.filter(\.isNumber)) ?? -999) - num) <= 40 }
        .map { dir + "/" + $0 }
    guard near.count > 1, let rows = (try? JSONSerialization.jsonObject(with: Data(run("/usr/bin/env", ["exiftool", "-j", "-n", "-q", "-SubSecDateTimeOriginal", "-DateTimeOriginal", "-SubSecTimeOriginal"] + near).utf8))) as? [[String: Any]] else { return [] }
    let fmt = DateFormatter(); fmt.dateFormat = "yyyy:MM:dd HH:mm:ss"; fmt.timeZone = TimeZone(identifier: "UTC")
    var times: [(String, Double)] = []
    for r in rows {
        guard let f = r["SourceFile"] as? String, let d = r["DateTimeOriginal"] as? String, let t = fmt.date(from: d) else { continue }
        let sub = Double("0." + String(describing: r["SubSecTimeOriginal"] ?? "0")) ?? 0
        times.append((f, t.timeIntervalSince1970 + sub))
    }
    times.sort { $0.1 < $1.1 }
    guard let i = times.firstIndex(where: { $0.0 == recipe.raw }) else { return [] }
    var lo = i, hi = i
    while lo > 0, times[lo].1 - times[lo - 1].1 <= 1.5 { lo -= 1 }
    while hi < times.count - 1, times[hi + 1].1 - times[hi].1 <= 1.5 { hi += 1 }
    let burst = times[lo...hi].map(\.0)
    guard burst.count > 1 else { return [] }
    var score: [String: (Double, String)] = [:]
    for line in run(binDir + "/bird_focus", burst, env: ["DECODER": "8"]).split(separator: "\n").dropFirst() {
        let c = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        guard c.count >= 7 else { continue }
        let v = Double(c[2]) ?? Double(c[1]) ?? 0
        score[dir + "/" + c[0]] = (v, c[6])
    }
    guard let refScore = score[recipe.raw]?.0, refScore > 0 else { return [] }
    let keep = burst.filter { $0 != recipe.raw && (score[$0]?.1.isEmpty ?? false) && (score[$0]?.0 ?? 0) >= 0.6 * refScore }
    return Array(keep.sorted { (score[$0]?.0 ?? 0) > (score[$1]?.0 ?? 0) }.prefix(7))
}

var stacked: CIImage? = nil
let wantStack = (recipe.stack?.isEmpty == false) || (recipe.stack_auto == true && iso >= (recipe.stack_min_iso ?? 3200))
if wantStack {
    let mates = recipe.stack ?? burstMates()
    if mates.isEmpty { report["stack"] = "no usable burst-mates" } else {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("stack_\(ProcessInfo.processInfo.processIdentifier)").path
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        _ = run(binDir + "/stack_burst", [dir, recipe.raw] + mates, env: ["RAW_OUT": "1", "NO_TIFF": "1"])
        let name = ((recipe.raw as NSString).lastPathComponent as NSString).deletingPathExtension
        if let js = try? JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name)_stack.json"))) as? [String: Any],
           let fpp = (js["frames_per_pixel"] as? NSNumber)?.doubleValue,
           let data = try? Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name)_stack.rgbah")), data.count == rb * Int(full.height) {
            var info = js; info["used"] = fpp >= 1.3
            report["stack"] = info
            if fpp >= 1.3 {
                let s = CIImage(bitmapData: data, bytesPerRow: rb, size: full.size, format: .RGBAh, colorSpace: linear)
                    .transformed(by: CGAffineTransform(translationX: full.minX, y: full.minY))
                stacked = s.settingAlphaOne(in: full)
            } else { warnings.append(String(format: "stack: only %.2f frames per pixel merged; used the single frame", fpp)) }
        } else { report["stack"] = "stacker failed" }
        try? FileManager.default.removeItem(atPath: dir)
    }
}
let buf = NSMutableData(length: rb * Int(full.height))!
ctx.render(stacked ?? raw.outputImage!, toBitmap: buf.mutableBytes, rowBytes: rb, bounds: full, format: .RGBAh, colorSpace: linear)
let develop = CIImage(bitmapData: buf as Data, bytesPerRow: rb, size: full.size, format: .RGBAh, colorSpace: linear)

func birdMask(_ img: CIImage, _ e: CGRect, af: CGRect?) throws -> CIImage {
    let win: CGRect = af.map { a in let s = max(1600, 6 * max(a.width, a.height)); return CGRect(x: a.midX - s / 2, y: a.midY - s / 2, width: s, height: s).intersection(e) } ?? e
    for w in [win, e] {
        let scale = min(1, 1024 / max(w.width, w.height))
        let handler = VNImageRequestHandler(ciImage: img.cropped(to: w).transformed(by: CGAffineTransform(translationX: -w.minX, y: -w.minY).concatenating(CGAffineTransform(scaleX: scale, y: scale))))
        let req = VNGenerateForegroundInstanceMaskRequest()
        try handler.perform([req])
        guard let obs = req.results?.first else { continue }
        var pick = obs.allInstances
        if let a = af {
            let lab = obs.instanceMask
            CVPixelBufferLockBaseAddress(lab, .readOnly)
            let lw = CVPixelBufferGetWidth(lab), lh = CVPixelBufferGetHeight(lab)
            let lx = Int((a.midX - w.minX) / w.width * CGFloat(lw)), ly = Int((1 - (a.midY - w.minY) / w.height) * CGFloat(lh))
            if lx >= 0, ly >= 0, lx < lw, ly < lh {
                let v = Int(CVPixelBufferGetBaseAddress(lab)!.assumingMemoryBound(to: UInt8.self)[ly * CVPixelBufferGetBytesPerRow(lab) + lx])
                if v > 0 { pick = IndexSet(integer: v) }
            }
            CVPixelBufferUnlockBaseAddress(lab, .readOnly)
        }
        let pb = try obs.generateScaledMaskForImage(forInstances: pick, from: handler)
        let m = CIImage(cvPixelBuffer: pb, options: [.colorSpace: NSNull()])
            .transformed(by: CGAffineTransform(scaleX: 1 / scale, y: 1 / scale).concatenating(CGAffineTransform(translationX: w.minX, y: w.minY)))
            .clampedToExtent().cropped(to: w)
        let mm = grey(m).composited(over: CIImage(color: .black)).cropped(to: e)

        if w != e {
            let r = readback(mm, w, scale: 64 / max(w.width, w.height), gamma: false)
            var edge: Float = 0
            for x in 0..<r.w { edge = max(edge, r.px[x * 4], r.px[((r.h - 1) * r.w + x) * 4]) }
            for y in 0..<r.h { edge = max(edge, r.px[(y * r.w) * 4], r.px[(y * r.w + r.w - 1) * 4]) }
            if edge > 0.3 { continue }
        }
        return mm
    }
    return CIImage(color: .black).cropped(to: e)
}
var maskFull = try birdMask(develop, full, af: afFull)
var maskSource = "foreground instance mask"
func pointSeedMask(_ img: CIImage, _ e: CGRect, af: CGRect) async throws -> CIImage? {
    let scale: CGFloat = min(1, 2048 / max(e.width, e.height))
    let small = img.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    guard let cgimg = ctx.createCGImage(small.applyingFilter("CILinearToSRGBToneCurve"), from: small.extent) else { return nil }
    var req = GenerateIterativeSegmentationRequest(seedPoint: NormalizedPoint(x: Double((af.midX - e.minX) / e.width), y: Double((af.midY - e.minY) / e.height)))
    req.qualityLevel = .accurate
    if "\(await req.assetStatus)" != "ready" { try await req.downloadAssets() }
    guard let obs = try await req.perform(on: cgimg) else { return nil }
    let m = CIImage(cgImage: try obs.cgImage, options: [.colorSpace: NSNull()])
    return grey(m.transformed(by: CGAffineTransform(scaleX: e.width / m.extent.width, y: e.height / m.extent.height))).composited(over: CIImage(color: .black)).cropped(to: e)
}
if let a = afFull {
    let probe = readback(maskFull, full, scale: 64 / full.width, gamma: false)
    if !probe.px.enumerated().contains(where: { $0.offset % 4 == 0 && $0.element > 0.5 }), let m = try await pointSeedMask(develop, full, af: a) {
        maskFull = m
        maskSource = "segmented from the camera's AF point (automatic mask found no bird): check it"
        warnings.append("bird mask came from the AF point, not the automatic subject mask: check it")
    }
}
report["mask_source"] = maskSource

let aScale: CGFloat = 1032 / full.width
let A = readback(develop, full, scale: aScale)
let M = readback(maskFull, full, scale: aScale, gamma: false)
var birdL: [Float] = [], bgL: [Float] = [], allL: [Float] = []
var bx0 = A.w, by0 = A.h, bx1 = 0, by1 = 0, birdN = 0
var massX: Float = 0
var hiWarm: [Float] = []
var neutralSum = [Float](repeating: 0, count: 3), neutralN = 0
var birdRGB = [Float](repeating: 0, count: 3)
var birdPix: [(Float, Float, Float, Float)] = []
for y in 0..<A.h { for x in 0..<A.w {
    let i = (y * A.w + x) * 4, l = lum(A.px, i), m = M.px[i]
    allL.append(l)
    if m > 0.8 {
        birdL.append(l); birdN += 1; massX += Float(x)
        bx0 = min(bx0, x); bx1 = max(bx1, x); by0 = min(by0, y); by1 = max(by1, y)
        for c in 0..<3 { birdRGB[c] += A.px[i + c] }
        birdPix.append((l, A.px[i], A.px[i + 1], A.px[i + 2]))

        let mx = max(A.px[i], A.px[i + 1], A.px[i + 2]), mn = min(A.px[i], A.px[i + 1], A.px[i + 2])
        if l > 0.2, mx < 0.97, (mx - mn) / max(mx, 1e-3) < 0.15 { for c in 0..<3 { neutralSum[c] += A.px[i + c] }; neutralN += 1 }
    } else if m < 0.05 { bgL.append(l) }
} }
let p05 = percentile(allL, 0.05), p95 = percentile(allL, 0.95), p80 = percentile(allL, 0.8)
for y in 0..<A.h { for x in 0..<A.w {
    let i = (y * A.w + x) * 4
    let mx = max(A.px[i], A.px[i + 1], A.px[i + 2]), mn = min(A.px[i], A.px[i + 1], A.px[i + 2])
    if M.px[i] < 0.05, lum(A.px, i) >= p80, mx > 0, (mx - mn) / mx < 0.35 { let s = A.px[i] + A.px[i + 1] + A.px[i + 2]; hiWarm.append((A.px[i] - A.px[i + 2]) / s) }
} }
let warmth = hiWarm.isEmpty ? 0 : hiWarm.reduce(0, +) / Float(hiWarm.count)
let birdMean = birdL.isEmpty ? 0 : birdL.reduce(0, +) / Float(birdL.count)
let bgMean = bgL.isEmpty ? 0 : bgL.reduce(0, +) / Float(bgL.count)
let range = p95 - p05
for c in 0..<3 { birdRGB[c] /= Float(max(1, birdN)) }

func gradMean(_ sel: (Float) -> Bool) -> Float {
    var s: Float = 0, n = 0
    for y in 1..<(A.h - 1) { for x in 1..<(A.w - 1) {
        let i = (y * A.w + x) * 4; if !sel(M.px[i]) { continue }
        s += abs(lum(A.px, i + 4) - lum(A.px, i - 4)) + abs(lum(A.px, i + A.w * 4) - lum(A.px, i - A.w * 4)); n += 1
    } }
    return n > 0 ? s / Float(n) : 0
}
let busy = gradMean { $0 < 0.05 } / max(gradMean { $0 > 0.8 }, 1e-4)

let lumaFull = develop.applyingFilter("CILinearToSRGBToneCurve").applyingFilter("CIColorControls", parameters: ["inputSaturation": 0])
let fine = lumaFull.applyingFilter("CIDifferenceBlendMode", parameters: ["inputBackgroundImage": lumaFull.clampedToExtent().applyingFilter("CIGaussianBlur", parameters: ["inputRadius": 1.0]).cropped(to: full)])
func noiseSigma() -> Float {
    let tile: CGFloat = 256
    var cands: [Float] = []
    for gy in stride(from: full.minY + tile, to: full.maxY - 2 * tile, by: full.height / 6) {
        for gx in stride(from: full.minX + tile, to: full.maxX - 2 * tile, by: full.width / 8) {
            let r = CGRect(x: gx, y: gy, width: tile, height: tile)
            let mk = readback(maskFull, r, scale: 1 / 16, gamma: false)
            if mk.px.enumerated().contains(where: { $0.offset % 4 == 0 && $0.element > 0.05 }) { continue }
            let f = readback(fine, r, scale: 1, gamma: false)
            var s: Float = 0; for k in stride(from: 0, to: f.px.count, by: 4) { s += f.px[k] * f.px[k] }
            cands.append(sqrt(s / Float(f.w * f.h)))
        }
    }
    cands.sort(); let q = cands.prefix(max(1, cands.count / 5))
    return q.isEmpty ? 0 : q.reduce(0, +) / Float(q.count)
}
let sigma = noiseSigma()

let birdBoxFull = birdN > 0 ? CGRect(x: CGFloat(bx0) / aScale, y: full.height - CGFloat(by1 + 1) / aScale,
                                      width: CGFloat(bx1 - bx0 + 1) / aScale, height: CGFloat(by1 - by0 + 1) / aScale) : full
func birdDetail() -> Float {
    let r = birdBoxFull.intersection(full).integral
    let f = readback(fine, r, scale: 1, gamma: false), m = readback(maskFull, r, scale: 1, gamma: false)
    var s: Float = 0, n: Float = 0
    for k in stride(from: 0, to: min(f.px.count, m.px.count), by: 4) where m.px[k] > 0.8 { s += f.px[k] * f.px[k]; n += 1 }
    return n > 0 ? sqrt(s / n) : 0
}
let birdFine = birdDetail()
let detailSNR = birdFine / max(sigma, 1e-5)
let sharpThreshold = Float(recipe.sharp_snr ?? 2.3)
let birdSharp = detailSNR >= sharpThreshold

report["analysis"] = ["bird_mean_L": birdMean, "background_mean_L": bgMean, "range_p05_p95": range, "highlight_warmth": warmth,
                      "background_busy_ratio": busy, "bird_range_p05_p95": percentile(birdL, 0.95) - percentile(birdL, 0.05), "noise_sigma": sigma, "bird_detail_rms": birdFine, "bird_detail_snr": detailSNR, "bird_sharp": birdSharp, "neutral_pixels": neutralN, "bird_pixels_analysis": birdN, "bird_neutral_share": Float(neutralN) / Float(max(1, birdN)),  "bird_white_share": Float(birdL.filter { $0 > 0.82 }.count) / Float(max(1, birdL.count))]

var L = Look()
var profiles: [String] = []

let birdRange = percentile(birdL, 0.95) - percentile(birdL, 0.05)
if birdRange < 0.45 { profiles.append("flat light"); L.env_cool_k = 0; L.shadow_cool_k = 0; L.bird_contrast = 1.05 }
if warmth > 0.06 && birdRange >= 0.45 { profiles.append("low warm sun"); L.env_highlight_warm_k = 150; L.shadow_cool_k = 300; L.wb_strength = 0.25 }
let bgHue = { () -> String in
    var s = [Float](repeating: 0, count: 3), n = 0
    for k in stride(from: 0, to: A.px.count, by: 4) where M.px[k] < 0.05 { for c in 0..<3 { s[c] += A.px[k + c] }; n += 1 }
    let r = s[0], g = s[1], b = s[2]
    return g > r * 1.05 && g > b ? "green" : b > r && b > g ? "blue" : "other"
}()
if (bgHue == "green" || bgHue == "blue") && birdRGB[0] > birdRGB[2] * 1.15 { profiles.append("warm subject on \(bgHue) background"); L.env_cool_k += 200 }
if busy > 0.6 { profiles.append("busy in-focus habitat"); L.burn = 0.4; L.global_vibrance = 0.04 }
else { profiles.append("soft background"); L.burn = 0.1 }
if bgMean < 0.22 && bgMean < birdMean { profiles.append("dark background: lift it"); L.env_ev_far = 0.25; L.env_cool_k = 0 }
let whiteShare = Float(birdL.filter { $0 > 0.82 }.count) / Float(max(1, birdL.count))
let neutralShare = Float(neutralN) / Float(max(1, birdN))
let brightest = birdPix.sorted { $0.0 > $1.0 }.prefix(max(1, birdPix.count / 10)).filter { max($0.1, $0.2, $0.3) < 0.97 }
let brightRGB = brightest.isEmpty ? [Float](repeating: 0, count: 3) : [brightest.map(\.1).reduce(0, +), brightest.map(\.2).reduce(0, +), brightest.map(\.3).reduce(0, +)].map { $0 / Float(brightest.count) }
let brightChroma = (brightRGB.max()! - brightRGB.min()!) / max(brightRGB.max()!, 1e-3)
report["bird_brightest_chroma"] = brightChroma
let birdChroma = birdPix.isEmpty ? 1 : birdPix.map { p -> Float in let mx = max(p.1, p.2, p.3); return mx > 1e-3 ? (mx - min(p.1, p.2, p.3)) / mx : 0 }.reduce(0, +) / Float(birdPix.count)
report["bird_mean_chroma"] = birdChroma
let castWhite = brightRGB[2] >= brightRGB[0] && birdChroma < 0.4 && neutralShare <= 0.35 && brightChroma > 0.12 && brightChroma < 0.3 && !profiles.contains("low warm sun") && brightest.count > 20
if castWhite {
    profiles.append("white plumage with a colour cast: neutralise it")
    neutralSum = brightRGB.map { $0 * Float(brightest.count) }; neutralN = brightest.count
    L.wb_strength = 0.9; L.bird_warm_k = 0
}
if neutralShare > 0.35 && !profiles.contains("low warm sun") { profiles.append("white or grey plumage: neutralise its cast"); L.wb_strength = 0.9; L.bird_warm_k = 0 }
if whiteShare > 0.12 { profiles.append("white plumage"); L.bird_ev = 0; L.eye_ev = 0.1; L.bird_highlights = 0.75; L.global_contrast = 1.03 }
if percentile(birdL, 0.99) > 0.93 { profiles.append("bright highlights on the bird"); L.bird_highlights = min(L.bird_highlights, 0.8); L.bird_ev = min(L.bird_ev, 0.05) }
if iso >= 10000 { profiles.append("high ISO"); L.bird_sharp = 0.2; L.bird_vibrance = 0.15 }
if stacked != nil { profiles.append("burst stack: sharpening the bird in place of RAW 9's"); L.bird_sharp = 1.8 }

if let o = recipe.look {
    var d = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(L))) as? [String: Any] ?? [:]
    for (k, v) in o { d[k] = v }
    if let data = try? JSONSerialization.data(withJSONObject: d), let l = try? JSONDecoder().decode(Look.self, from: data) { L = l }
}
if birdN == 0 { warnings.append("no bird found: nothing bird-specific was applied and the frame was left uncropped") }
report["profiles"] = profiles
report["look"] = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(L))) ?? [:]

let destinationPx: [String: Double] = ["web": 2048, "contest": 3000, "print_8x12": 3600, "print_12x18": 5400, "print_16x24": 7200, "print_20x30": 9000]
let destination = recipe.destination ?? "web"
let minLong = CGFloat(recipe.min_long_px ?? destinationPx[destination] ?? 2048)
let maxEffISO = recipe.max_effective_iso ?? 6400
let fullArea = full.width * full.height
var crop: CGRect
var cropNotes: [String] = []
if let c = recipe.crop {
    crop = CGRect(x: c[0], y: full.height - c[1] - c[3], width: c[2], height: c[3])
    cropNotes.append("crop from recipe")
} else if recipe.auto_crop ?? true {
    let lowNoiseRoom = iso <= 0 || iso <= maxEffISO * 0.64
    let ratioName = recipe.ratio ?? (birdBoxFull.height > birdBoxFull.width * 1.0 && lowNoiseRoom ? "4:5" : "3:2")
    let ratio: CGFloat = { switch ratioName { case "4:5": return 4.0 / 5; case "1:1": return 1; default: return 3.0 / 2 } }()

    let birdArea = CGFloat(birdN) / (aScale * aScale)
    var w = sqrt(birdArea / 0.30 * ratio)
    w = max(w, birdBoxFull.width * 1.25, birdBoxFull.height * 1.25 * ratio)
    if iso > 0 { w = max(w, sqrt(fullArea * CGFloat(iso / maxEffISO) * ratio)) }
    w = max(w, ratio >= 1 ? minLong : minLong * ratio)
    if !birdSharp { w = max(w, full.width * 0.667, full.height * 0.667 * ratio) }
    w = min(w, full.width, full.height * ratio)
    let h = w / ratio

    let eye = afFull.map { CGPoint(x: $0.midX, y: $0.midY) } ?? CGPoint(x: birdBoxFull.midX, y: birdBoxFull.maxY - birdBoxFull.height * 0.2)
    let centroidX = CGFloat(massX / Float(max(1, birdN))) / aScale
    let facesRight = eye.x >= centroidX
    var x = eye.x - (facesRight ? w / 3 : 2 * w / 3)
    var y = eye.y - 2 * h / 3

    let mx = 0.04 * w, my = 0.04 * h
    x = min(max(x, birdBoxFull.maxX + mx - w), birdBoxFull.minX - mx)
    y = min(max(y, birdBoxFull.maxY + my - h), birdBoxFull.minY - my)
    x = min(max(x, full.minX), full.maxX - w); y = min(max(y, full.minY), full.maxY - h)
    crop = CGRect(x: x, y: y, width: w, height: h).integral
    cropNotes.append("auto crop \(ratioName), bird \(Int(100 * birdArea / (w * h)))% of crop, \(facesRight ? "faces right" : "faces left")")
} else { crop = full; cropNotes.append("uncropped") }

let cropFactor = sqrt(fullArea / (crop.width * crop.height))
let effISO = iso * Double(cropFactor * cropFactor)
let longPx = max(crop.width, crop.height)
if iso > 0 && cropFactor > 1.05 && effISO > maxEffISO { warnings.append(String(format: "noise: effective ISO %.0f after a %.2fx crop (limit %.0f)", effISO, cropFactor, maxEffISO)) }
if longPx < minLong { warnings.append(String(format: "resolution: %.0f px long side, %@ needs %.0f", longPx, destination, minLong)) }
if cropFactor > 1.5 && !birdSharp { warnings.append(String(format: "deep %.2fx crop on a bird that is not sharp (detail/noise %.1f)", cropFactor, detailSNR)) }
report["crop_tests"] = ["crop_factor": cropFactor, "effective_iso": effISO, "max_effective_iso": maxEffISO, "long_side_px": longPx,
                        "destination": destination, "min_long_px": minLong, "bird_detail_snr": detailSNR, "bird_sharp": birdSharp,
                        "largest_print_long_edge_in_at_300ppi": Double(longPx) / 300]
let birdAtFrameEdge = birdBoxFull.minX <= full.minX + 8 || birdBoxFull.minY <= full.minY + 8 || birdBoxFull.maxX >= full.maxX - 8 || birdBoxFull.maxY >= full.maxY - 8
if birdAtFrameEdge { cropNotes.append("bird already cut by the camera frame") }
if !birdAtFrameEdge && !crop.insetBy(dx: crop.width * 0.01, dy: crop.height * 0.01).contains(birdBoxFull) { cropNotes.append("bird touches or crosses the crop edge"); warnings.append("bird touches the crop edge") }
let birdFracCrop = Double(CGFloat(birdN) / (aScale * aScale) / (crop.width * crop.height))
report["crop"] = ["rect_top_left": [crop.minX, full.height - crop.maxY, crop.width, crop.height], "bird_area_pct": birdFracCrop * 100, "notes": cropNotes]

let t = CGAffineTransform(translationX: -crop.minX, y: -crop.minY)
let base0 = develop.cropped(to: crop).transformed(by: t)
let mask = maskFull.cropped(to: crop).transformed(by: t)
let e = base0.extent
let af = afFull.map { $0.applying(t) }

var base = base0
if neutralN > max(50, birdN / 200) {
    let n = neutralSum.map { v -> Float in let g = v / Float(neutralN); return g <= 0.04045 ? g / 12.92 : powf((g + 0.055) / 1.055, 2.4) }
    let g = (n[0] + n[1] + n[2]) / 3
    let gains = n.map { min(1.25, max(0.8, 1 + Float(L.wb_strength) * (g / max($0, 1e-3) - 1))) }
    base = base.applyingFilter("CIColorMatrix", parameters: ["inputRVector": CIVector(x: CGFloat(gains[0]), y: 0, z: 0, w: 0),
        "inputGVector": CIVector(x: 0, y: CGFloat(gains[1]), z: 0, w: 0), "inputBVector": CIVector(x: 0, y: 0, z: CGFloat(gains[2]), w: 0)])
    report["white_balance"] = ["neutral_rgb": n, "gains": gains]
} else { report["white_balance"] = "no neutral feathers found; left as shot" }

let q: CGFloat = 0.25
let birdSize = sqrt(Double(CGFloat(birdN) / (aScale * aScale)))
let maxDist = max(40, L.falloff * birdSize * Double(q))
let small = mask.transformed(by: CGAffineTransform(scaleX: q, y: q))
let dist = grey(small.applyingFilter("CIDistanceGradientFromRedMask", parameters: ["inputMaximumDistance": maxDist])
    .cropped(to: small.extent).transformed(by: CGAffineTransform(scaleX: 1 / q, y: 1 / q)).clampedToExtent().cropped(to: e))
let ramp = dist.applyingFilter("CIToneCurve", parameters: ["inputPoint0": CIVector(x: 0, y: 0), "inputPoint1": CIVector(x: 0.25, y: 0.16),
    "inputPoint2": CIVector(x: 0.5, y: 0.5), "inputPoint3": CIVector(x: 0.75, y: 0.84), "inputPoint4": CIVector(x: 1, y: 1)])
let notBird = mask.applyingFilter("CIColorInvert")
let envW = mult(ramp, notBird).cropped(to: e)
func blend(_ a: CIImage, over b: CIImage, _ m: CIImage) -> CIImage { a.applyingFilter("CIBlendWithMask", parameters: ["inputBackgroundImage": b, "inputMaskImage": m]).cropped(to: e) }
var img = blend(temperature(base.applyingFilter("CIExposureAdjust", parameters: ["inputEV": L.env_ev_far]), warmerBy: -L.env_cool_k), over: base, envW)

if L.shadow_cool_k != 0 {
    let shadows = mult(lumaKey(img, lo: 0.15, hi: 0.45).applyingFilter("CIColorInvert"), notBird)
    img = blend(temperature(img, warmerBy: -L.shadow_cool_k), over: img, shadows)
}
if L.env_highlight_warm_k != 0 { img = blend(temperature(img, warmerBy: L.env_highlight_warm_k), over: img, mult(lumaKey(img, lo: 0.55, hi: 0.85), notBird)) }

let bgBright = Double(percentile(bgL, 0.85)), bgTop = Double(percentile(bgL, 0.98))
let burnW = mult(mult(lumaKey(img, lo: bgBright, hi: max(bgTop, bgBright + 0.05)), envW), notBird).cropped(to: e)
img = blend(img.applyingFilter("CIExposureAdjust", parameters: ["inputEV": -L.burn]).applyingFilter("CIColorControls", parameters: ["inputSaturation": 0.85, "inputContrast": 1, "inputBrightness": 0]), over: img, burnW)

let noiseLevel = Double(min(0.06, max(0.004, 2.0 * sigma))) * L.bg_denoise
let bgDenoised = img.clampedToExtent().applyingFilter("CINoiseReduction", parameters: ["inputNoiseLevel": noiseLevel, "inputSharpness": 0.0]).cropped(to: e)
img = blend(bgDenoised, over: img, notBird)
report["denoise"] = ["background_noise_level": noiseLevel, "bird": stacked == nil ? "RAW 9 default" : "burst stack"]

var bird = temperature(img.applyingFilter("CIExposureAdjust", parameters: ["inputEV": L.bird_ev]), warmerBy: L.bird_warm_k)
bird = bird.applyingFilter("CIHighlightShadowAdjust", parameters: ["inputShadowAmount": L.shadow_lift, "inputHighlightAmount": L.bird_highlights, "inputRadius": 4])
if L.bird_contrast != 1 {
    bird = bird.applyingFilter("CILinearToSRGBToneCurve").applyingFilter("CIColorControls", parameters: ["inputContrast": L.bird_contrast, "inputSaturation": 1, "inputBrightness": 0]).applyingFilter("CISRGBToneCurveToLinear")
}
if let a = af, L.eye_ev != 0 {
    let r = max(a.width, a.height) * 0.6
    let eyeMask = grey(CIImage.empty().applyingFilter("CIRadialGradient", parameters: ["inputCenter": CIVector(x: a.midX, y: a.midY), "inputRadius0": r * 0.3, "inputRadius1": r,
        "inputColor0": CIColor(red: 1, green: 1, blue: 1), "inputColor1": CIColor(red: 0, green: 0, blue: 0)]).cropped(to: e))
    let eyeIn = mult(mult(eyeMask, mask), lumaKey(bird, lo: 0.2, hi: 0.45)).cropped(to: e)
    bird = blend(bird.applyingFilter("CIExposureAdjust", parameters: ["inputEV": L.eye_ev]), over: bird, eyeIn)
}
bird = bird.clampedToExtent().applyingFilter("CISharpenLuminance", parameters: ["inputSharpness": L.bird_sharp, "inputRadius": 1.2]).cropped(to: e)
bird = bird.applyingFilter("CIVibrance", parameters: ["inputAmount": L.bird_vibrance])
img = blend(bird, over: img, mask)

img = img.applyingFilter("CILinearToSRGBToneCurve")
    .applyingFilter("CIColorControls", parameters: ["inputContrast": L.global_contrast, "inputSaturation": 1, "inputBrightness": 0])
img = hotKernel.apply(extent: e, arguments: [img.applyingFilter("CIVibrance", parameters: ["inputAmount": L.global_vibrance]), Float(L.hot_colour)])!
    .applyingFilter("CISRGBToneCurveToLinear").cropped(to: e)

let cScale: CGFloat = min(1, 1600 / e.width)
let O = readback(base0, e, scale: cScale), Edt = readback(img, e, scale: cScale), Mk = readback(mask, e, scale: cScale, gamma: false)

let hard = mask.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": 0.5]).cropped(to: e)
func ring(_ r: Double) -> (px: [Float], w: Int, h: Int) { readback(hard.applyingFilter("CIMorphologyMaximum", parameters: ["inputRadius": r]).cropped(to: e), e, scale: cScale, gamma: false) }
let d4 = ring(4), d12 = ring(12), d16 = ring(16), d40 = ring(40)
var clipBird = 0, birdC = 0, clipAll = 0, clipBefore = 0
var nearO: [Float] = [], nearE: [Float] = [], farO: [Float] = [], farE: [Float] = [], bO: [Float] = [], bE: [Float] = []
var histE: [Float] = []
for k in stride(from: 0, to: min(Edt.px.count, Mk.px.count, d4.px.count, d40.px.count), by: 4) {
    let m = Mk.px[k], le = lum(Edt.px, k), lo = lum(O.px, k)
    histE.append(le)
    let clipped = max(Edt.px[k], Edt.px[k + 1], Edt.px[k + 2]) >= 0.995
    if clipped { clipAll += 1 }
    if m > 0.8 { birdC += 1; if clipped { clipBird += 1 }; if max(O.px[k], O.px[k + 1], O.px[k + 2]) >= 0.995 { clipBefore += 1 }; bO.append(lo); bE.append(le) }
    else if d12.px[k] > 0.5 && d4.px[k] < 0.5 { nearO.append(lo); nearE.append(le) }
    else if d40.px[k] > 0.5 && d16.px[k] < 0.5 { farO.append(lo); farE.append(le) }
}
func mean(_ v: [Float]) -> Float { v.isEmpty ? 0 : v.reduce(0, +) / Float(v.count) }
let clipBirdPct = Double(clipBird) / Double(max(1, birdC)) * 100
let clipBeforePct = Double(clipBefore) / Double(max(1, birdC)) * 100
if clipBirdPct > 0.5 && clipBirdPct > clipBeforePct + 0.25 { warnings.append(String(format: "clipping on the bird %.2f%% (%.2f%% already clipped before the edit)", clipBirdPct, clipBeforePct)) }

let rimO = mean(nearO) / max(mean(farO), 1e-3), rimE = mean(nearE) / max(mean(farE), 1e-3)
let haloMeasured = !nearO.isEmpty && !farO.isEmpty
let halo = haloMeasured ? Double(rimE / max(rimO, 1e-3) - 1) : 0
if !haloMeasured { report["halo"] = "not measurable: no habitat ring around the bird" }
if haloMeasured && abs(halo) > 0.06 { warnings.append(String(format: "possible %@ halo (rim %+.0f%% vs nearby habitat)", halo > 0 ? "bright" : "dark", halo * 100)) }

let stickO = log2(max(mean(bO), 1e-3) / max(mean(nearO + farO), 1e-3)), stickE = log2(max(mean(bE), 1e-3) / max(mean(nearE + farE), 1e-3))
let sticker = Double(abs(stickE) - abs(stickO))
if sticker > 0.35 { warnings.append(String(format: "sticker risk: bird/habitat contrast +%.2f stops", sticker)) }

if let refPath = recipe.reference, let ref = CIImage(contentsOf: URL(fileURLWithPath: refPath)) {
    let R = readback(ref, ref.extent, scale: min(1, 512 / ref.extent.width))
    var rr = [Float](repeating: 0, count: 3); var n = 0
    for k in stride(from: 0, to: R.px.count, by: 4) { for c in 0..<3 { rr[c] += R.px[k + c] }; n += 1 }
    var bb = [Float](repeating: 0, count: 3); var nb = 0
    for k in stride(from: 0, to: Edt.px.count, by: 4) where Mk.px[k] > 0.8 { for c in 0..<3 { bb[c] += Edt.px[k + c] }; nb += 1 }
    let refC = rr.map { $0 / Float(max(1, n)) }, birdCc = bb.map { $0 / Float(max(1, nb)) }
    let hueShift = zip(refC.map { $0 / max(refC.reduce(0, +), 1e-3) }, birdCc.map { $0 / max(birdCc.reduce(0, +), 1e-3) }).map { abs($0 - $1) }.max() ?? 0
    report["colour_vs_reference"] = ["reference_rgb": refC, "bird_rgb": birdCc, "max_channel_share_diff": hueShift]
    if hueShift > 0.05 { warnings.append(String(format: "plumage colour differs from the reference (channel share %.0f%%)", hueShift * 100)) }
} else { report["colour_vs_reference"] = "no reference given (check plumage against a reference by eye)" }
report["checks"] = ["clipped_on_bird_pct": clipBirdPct, "clipped_on_bird_before_edit_pct": clipBeforePct, "clipped_whole_pct": Double(clipAll) / Double(max(1, histE.count)) * 100,
                    "halo_rim_change": halo, "sticker_extra_stops": sticker,
                    "histogram_p01_p50_p99": [percentile(histE, 0.01), percentile(histE, 0.5), percentile(histE, 0.99)],
                    "bird_vs_habitat_stops_before_after": [stickO, stickE]]
report["warnings"] = warnings

var out = img
if let w = recipe.out_width, w != Double(e.width) {
    let s = w / Double(e.width)
    out = out.applyingFilter("CILanczosScaleTransform", parameters: ["inputScale": s, "inputAspectRatio": 1])
        .cropped(to: CGRect(x: 0, y: 0, width: w.rounded(), height: (Double(e.height) * s).rounded()))
}
let q92 = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
try ctx.writeJPEGRepresentation(of: out, to: URL(fileURLWithPath: recipe.out + ".jpg"), colorSpace: CGColorSpace(name: CGColorSpace.displayP3)!, options: [q92: 0.92])
try ctx.writePNGRepresentation(of: mask, to: URL(fileURLWithPath: recipe.out + "_mask.png"), format: .L8, colorSpace: CGColorSpace(name: CGColorSpace.linearGray)!)
let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
try json.write(to: URL(fileURLWithPath: recipe.out + "_report.json"))
print("\((recipe.raw as NSString).lastPathComponent): profiles \(profiles.joined(separator: ", ")) | \(cropNotes.joined(separator: "; ")) | " +
      String(format: "noise σ %.4f | clip %.2f%% (was %.2f%%) | halo %+.0f%% | sticker %+.2f st", sigma, clipBirdPct, clipBeforePct, halo * 100, sticker) +
      (warnings.isEmpty ? " | checks pass" : " | WARN: " + warnings.joined(separator: "; ")))
