import CoreImage
import CoreML
import Vision
import Accelerate
import Metal
import MetalPerformanceShadersGraph

let env = ProcessInfo.processInfo.environment

let memoryMB = Int(env["CI_MEMORY_MB"] ?? "") ?? 4096
let workers = max(1, Int(env["WORKERS"] ?? "") ?? 8)
let useRaw8 = env["DECODER"] == "8"
let visionDevice = env["VISION_DEVICE"] ?? "mix"
let timing = env["TIMING"] != nil

let mtl = MTLCreateSystemDefaultDevice()!
func makeContext() -> CIContext { CIContext(mtlDevice: mtl, options: [.cacheIntermediates: false, .workingColorSpace: NSNull(), .memoryTarget: memoryMB]) }
let sharedContexts = env["SHARED_CONTEXT"] != nil
let shared = makeContext()
let contextKey = "bird_focus.ctx"
var ctx: CIContext {
    if sharedContexts { return shared }
    if let c = Thread.current.threadDictionary[contextKey] as? CIContext { return c }
    let c = makeContext(); Thread.current.threadDictionary[contextKey] = c; return c
}

let packKernels = try! CIKernel.kernels(withMetalString: """
#include <CoreImage/CoreImage.h>
extern "C" { namespace coreimage {
[[ stitchable ]] float4 packBands(sample_t l, sample_t g1, sample_t g8, sample_t g24, sample_t m, sample_t o) {
    float f = l.r - g1.r, c = g8.r - g24.r;
    return float4(f * f * m.r, c * c * m.r, m.r, o.r);
}
[[ stitchable ]] float4 fineBand(sample_t l, sample_t g1) {
    float f = l.r - g1.r;
    return float4(f * f, 0.0, 0.0, 0.0);
}
}}
""")
let packKernel = packKernels.first { $0.name == "packBands" } as! CIColorKernel
let fineKernel = packKernels.first { $0.name == "fineBand" } as! CIColorKernel

let block = 4
final class BlockSums {
    let graph = MPSGraph()
    let queue = mtl.makeCommandQueue()!
    let input: MPSGraphTensor
    let output: MPSGraphTensor
    init() {
        input = graph.placeholder(shape: [1, -1, -1, 4], dataType: .float32, name: nil)
        let d = MPSGraphPooling2DOpDescriptor(kernelWidth: block, kernelHeight: block, strideInX: block, strideInY: block, paddingStyle: .TF_VALID, dataLayout: .NHWC)!
        output = graph.multiplication(graph.avgPooling2D(withSourceTensor: input, descriptor: d, name: nil), graph.constant(Double(block * block), dataType: .float32), name: nil)
    }
    func run(_ image: CIImage, bounds: CGRect, context: CIContext) -> (sums: [Float], bw: Int, bh: Int) {
        let W = Int(bounds.width), H = Int(bounds.height)
        let Wp = (W + 63) / 64 * 64, Hp = (H + block - 1) / block * block
        let buf = mtl.makeBuffer(length: Wp * Hp * 16, options: .storageModeShared)!
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: W, height: H, mipmapped: false)
        td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .shared
        let tex = buf.makeTexture(descriptor: td, offset: 0, bytesPerRow: Wp * 16)!
        let dest = CIRenderDestination(mtlTexture: tex, commandBuffer: nil)
        dest.colorSpace = nil
        _ = try? context.startTask(toRender: image.transformed(by: CGAffineTransform(translationX: -bounds.minX, y: -bounds.minY)), from: CGRect(x: 0, y: 0, width: W, height: H), to: dest, at: .zero).waitUntilCompleted()
        let data = MPSGraphTensorData(buf, shape: [1, NSNumber(value: Hp), NSNumber(value: Wp), 4], dataType: .float32)
        let res = graph.run(with: queue, feeds: [input: data], targetTensors: [output], targetOperations: nil)[output]!
        let bw = Wp / block, bh = Hp / block
        var sums = [Float](repeating: 0, count: bw * bh * 4)
        res.mpsndarray().readBytes(&sums, strideBytes: nil)
        return (sums, bw, bh)
    }
}
let sumsKey = "bird_focus.sums"
var blockSums: BlockSums {
    if let b = Thread.current.threadDictionary[sumsKey] as? BlockSums { return b }
    let b = BlockSums(); Thread.current.threadDictionary[sumsKey] = b; return b
}

func afBoxes(_ paths: [String]) -> [String: [Double]] {
    let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    p.arguments = ["exiftool", "-j", "-n", "-q", "-AFAreaXPosition", "-AFAreaYPosition", "-AFAreaWidth", "-AFAreaHeight", "-AFImageWidth", "-AFImageHeight", "-Orientation"] + paths
    let pipe = Pipe(); p.standardOutput = pipe; try? p.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile(); p.waitUntilExit()
    guard let rows = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] else { return [:] }
    var out: [String: [Double]] = [:]
    for r in rows {
        let keys = ["AFAreaXPosition", "AFAreaYPosition", "AFAreaWidth", "AFAreaHeight", "AFImageWidth", "AFImageHeight"]
        let v = keys.compactMap { (r[$0] as? NSNumber)?.doubleValue }
        if let f = r["SourceFile"] as? String, v.count == 6, v[2] > 0 { out[f] = v + [(r["Orientation"] as? NSNumber)?.doubleValue ?? 1] }
    }
    return out
}

let gpuDevice: MLComputeDevice? = MLComputeDevice.allComputeDevices.first { if case .gpu = $0 { return true }; return false }
let aneDevice: MLComputeDevice? = MLComputeDevice.allComputeDevices.first { if case .neuralEngine = $0 { return true }; return false }

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

struct Pick { let obs: VNInstanceMaskObservation; let handler: VNImageRequestHandler; let instance: Int; let onAF: Bool; let touchesWindow: Bool; let scale: CGFloat }

func pickSubject(_ img: CIImage, win: CGRect, af: CGRect, device: MLComputeDevice?) throws -> Pick? {
    let scale = min(1, 1024 / max(win.width, win.height))
    let view = img.cropped(to: win).transformed(by: CGAffineTransform(translationX: -win.minX, y: -win.minY).concatenating(CGAffineTransform(scaleX: scale, y: scale)))
    let handler = VNImageRequestHandler(ciImage: view)
    let req = VNGenerateForegroundInstanceMaskRequest()
    if let d = device { try? req.setComputeDevice(d, for: .main) }
    try handler.perform([req])
    guard let obs = req.results?.first, !obs.allInstances.isEmpty else { return nil }
    let lab = obs.instanceMask
    CVPixelBufferLockBaseAddress(lab, .readOnly); defer { CVPixelBufferUnlockBaseAddress(lab, .readOnly) }
    let w = CVPixelBufferGetWidth(lab), h = CVPixelBufferGetHeight(lab), rb = CVPixelBufferGetBytesPerRow(lab)
    let base = CVPixelBufferGetBaseAddress(lab)!.assumingMemoryBound(to: UInt8.self)
    let ax = Double((af.midX - win.minX) / win.width) * Double(w), ay = (1 - Double((af.midY - win.minY) / win.height)) * Double(h)
    var area = [Int: Int](), dist = [Int: Double](), edges = [Int: Set<Int>]()
    for y in 0..<h { for x in 0..<w {
        let v = Int(base[y * rb + x]); if v == 0 { continue }
        area[v, default: 0] += 1
        dist[v] = min(dist[v] ?? .infinity, hypot(Double(x) + 0.5 - ax, Double(y) + 0.5 - ay))
        if x == 0 { edges[v, default: []].insert(0) }; if x == w - 1 { edges[v, default: []].insert(1) }
        if y == 0 { edges[v, default: []].insert(2) }; if y == h - 1 { edges[v, default: []].insert(3) }
    } }
    let under: Int? = { let x = Int(ax), y = Int(ay); guard x >= 0, y >= 0, x < w, y < h else { return nil }
        let v = Int(base[y * rb + x]); return v > 0 ? v : nil }()
    let chosen: Int
    if let u = under { chosen = u }
    else if let near = area.keys.filter({ (edges[$0]?.count ?? 0) < 2 }).min(by: { dist[$0]! < dist[$1]! }) { chosen = near }
    else { chosen = area.max(by: { $0.value < $1.value })!.key }
    return Pick(obs: obs, handler: handler, instance: chosen, onAF: under != nil, touchesWindow: !(edges[chosen] ?? []).isEmpty, scale: scale)
}

func score(path: String, index: Int, af afBox: [Double]?) -> String {
    let name = (path as NSString).lastPathComponent
    var times: [String] = []; var mark = Date()
    func lap(_ l: String) { if timing { times.append("\(l) \(String(format: "%.2f", Date().timeIntervalSince(mark)))s"); mark = Date() } }
    defer { if timing { FileHandle.standardError.write("\(name): \(times.joined(separator: ", "))\n".data(using: .utf8)!) } }
    guard let raw = CIRAWFilter(imageURL: URL(fileURLWithPath: path)) else { return "\(name)\t\t\t\t\t\tcannot open" }
    if !useRaw8, raw.supportedDecoderVersions.contains(.version9) { raw.decoderVersion = .version9 } else { raw.decoderVersion = .version8 }
    if raw.isSharpnessSupported { raw.sharpnessAmount = 0 }

    let decoded = raw.outputImage!, e = decoded.extent
    let rowBytes = Int(e.width) * 8
    let data = NSMutableData(length: rowBytes * Int(e.height))!
    ctx.render(decoded, toBitmap: data.mutableBytes, rowBytes: rowBytes, bounds: e, format: .RGBAh, colorSpace: nil)
    let img = CIImage(bitmapData: data as Data, bytesPerRow: rowBytes, size: e.size, format: .RGBAh, colorSpace: nil)
    lap("decode")
    var flags: [String] = []

    var af = CGRect(x: e.midX - 150, y: e.midY - 150, width: 300, height: 300)
    if let b = afBox {
        af = displayAF(b[0], b[1], b[2], b[3], b[4], b[5], orientation: Int(b[6]), in: e)
    } else { flags.append("no AF box") }

    let device: MLComputeDevice? = visionDevice == "gpu" ? gpuDevice : visionDevice == "ane" ? aneDevice : (index % 2 == 0 ? aneDevice : gpuDevice)
    var side = max(1600, 6 * max(af.width, af.height))
    var pick: Pick?, win = e
    do {
        while true {
            win = CGRect(x: af.midX - side / 2, y: af.midY - side / 2, width: side, height: side).intersection(e).integral
            pick = try pickSubject(img, win: win, af: af, device: device)
            guard let p = pick, p.touchesWindow, win != e else { break }
            side *= 2
        }
    } catch { return "\(name)\t\t\t\t\t\tvision error: \(error.localizedDescription)" }
    lap("vision")
    guard let p = pick else { return "\(name)\t\t\t\t\t\t" + (flags + ["no subject"]).joined(separator: "; ") }
    if !p.onAF { flags.append("AF point off bird") }

    guard let buf = try? p.obs.generateScaledMaskForImage(forInstances: IndexSet(integer: p.instance), from: p.handler) else { return "\(name)\t\t\t\t\t\tmask error" }
    let mask = CIImage(cvPixelBuffer: buf, options: [.colorSpace: NSNull()])
        .transformed(by: CGAffineTransform(scaleX: 1 / p.scale, y: 1 / p.scale).concatenating(CGAffineTransform(translationX: win.minX, y: win.minY)))
        .clampedToExtent().cropped(to: win)
    let inner = mask.applyingFilter("CIMorphologyMinimum", parameters: ["inputRadius": 4]).cropped(to: win)

    let luma = img.cropped(to: win).applyingFilter("CILinearToSRGBToneCurve").applyingFilter("CIColorControls", parameters: ["inputSaturation": 0])
    func blur(_ sigma: Double) -> CIImage { luma.clampedToExtent().applyingFilter("CIGaussianBlur", parameters: ["inputRadius": sigma]).cropped(to: win) }
    let packed = packKernel.apply(extent: win, arguments: [luma, blur(1), blur(8), blur(24), inner, mask])!
    let fine = fineKernel.apply(extent: win, arguments: [luma, blur(1)])!
    let W = Int(win.width), H = Int(win.height)
    let a = blockSums.run(packed, bounds: win, context: ctx)
    let b = blockSums.run(fine, bounds: win, context: ctx)
    let bw = a.bw, bh = a.bh
    var integ = [Double](repeating: 0, count: (bw + 1) * (bh + 1) * 5)
    for y in 0..<bh { for x in 0..<bw { for k in 0..<5 {
        let v = Double(k < 4 ? a.sums[(y * bw + x) * 4 + k] : b.sums[(y * bw + x) * 4])
        let i = ((y + 1) * (bw + 1) + x + 1) * 5 + k
        integ[i] = v + integ[i - (bw + 1) * 5] + integ[i - 5] - integ[i - (bw + 2) * 5]
    } } }
    lap("bands")

    func bounds(_ r: CGRect) -> (x0: Int, x1: Int, y0: Int, y1: Int)? {
        let rr = r.intersection(win); if rr.isNull || rr.width < 1 || rr.height < 1 { return nil }
        let px = (max(0, Int(rr.minX - win.minX)), min(W, Int(rr.maxX - win.minX)), max(0, Int(rr.minY - win.minY)), min(H, Int(rr.maxY - win.minY)))
        let x0 = min(bw - 1, (px.0 + block / 2) / block), y0 = min(bh - 1, (px.2 + block / 2) / block)
        return (x0, max(x0 + 1, min(bw, (px.1 + block / 2) / block)), y0, max(y0 + 1, min(bh, (px.3 + block / 2) / block)))
    }
    func rect(_ b: (x0: Int, x1: Int, y0: Int, y1: Int), _ k: Int) -> Float {
        let at = { (x: Int, y: Int) in integ[(y * (bw + 1) + x) * 5 + k] }
        return Float(at(b.x1, b.y1) - at(b.x0, b.y1) - at(b.x1, b.y0) + at(b.x0, b.y0))
    }
    func area(_ b: (x0: Int, x1: Int, y0: Int, y1: Int)) -> Float { Float((b.x1 - b.x0) * (b.y1 - b.y0) * block * block) }
    func maskMean(_ r: CGRect) -> Float {
        guard let b = bounds(r) else { return 0 }
        return rect(b, 3) / area(b)
    }
    func stats(_ r: CGRect, weighted: Bool) -> (fine: Float, coarse: Float, cover: Float) {
        guard let b = bounds(r) else { return (0, 0, 0) }
        let n = area(b)
        return ((weighted ? rect(b, 0) : rect(b, 4)) / n, rect(b, 1) / n, rect(b, 2) / n)
    }

    let strips = [CGRect(x: e.minX, y: e.minY, width: 8, height: e.height), CGRect(x: e.maxX - 8, y: e.minY, width: 8, height: e.height),
                  CGRect(x: e.minX, y: e.minY, width: e.width, height: 8), CGRect(x: e.minX, y: e.maxY - 8, width: e.width, height: 8)]
    if strips.contains(where: { maskMean($0) > 0.05 }) { flags.append("cut off by frame") }

    let n = 6
    var tiles: [Float] = []
    for i in 0..<n { for j in 0..<n {
        let r = CGRect(x: win.minX + win.width * CGFloat(i) / CGFloat(n), y: win.minY + win.height * CGFloat(j) / CGFloat(n), width: win.width / CGFloat(n), height: win.height / CGFloat(n))
        if maskMean(r) < 0.01 { tiles.append(stats(r, weighted: false).fine) }
    } }
    tiles.sort()
    let quiet = tiles.prefix(max(1, tiles.count / 5))
    let noise = quiet.isEmpty ? 0 : quiet.reduce(0, +) / Float(quiet.count)
    if tiles.isEmpty { flags.append("no background for noise") }

    func measure(_ r: CGRect, minCover: Float) -> (focus: Float, detail: Float)? {
        let s = stats(r, weighted: true); guard s.cover > minCover else { return nil }
        let fine = max(s.fine / s.cover - noise, 0), coarse = s.coarse / s.cover
        return (100 * fine / max(coarse, 1e-7), fine * 1e4)
    }
    let bird = measure(win, minCover: 0.002)
    let eye = measure(af, minCover: 0.05)
    let birdPx = Int(Double(maskMean(win)) * win.width * win.height)
    lap("stats")
    func s(_ v: Float?) -> String { v.map { String(format: "%.3f", $0) } ?? "" }
    return "\(name)\t\(s(bird?.focus))\t\(s(eye?.focus))\t\(s(bird?.detail))\t\(String(format: "%.3f", noise * 1e4))\t\(birdPx)\t\(flags.joined(separator: "; "))"
}

let paths = Array(CommandLine.arguments.dropFirst())
let boxes = afBoxes(paths)
var results = [String](repeating: "", count: paths.count)
let lock = NSLock()
var next = 0

DispatchQueue.concurrentPerform(iterations: min(workers, paths.count)) { _ in
    while true {
        lock.lock(); let i = next; next += 1; lock.unlock()
        if i >= paths.count { break }
        let line = autoreleasepool { score(path: paths[i], index: i, af: boxes[paths[i]]) }
        lock.lock(); results[i] = line; lock.unlock()
    }
}
print("file\tfocus\teye_focus\tdetail\tnoise\tbird_px\tflags")
results.forEach { print($0) }
