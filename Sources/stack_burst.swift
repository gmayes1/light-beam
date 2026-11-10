import CoreImage
import Vision
import Metal
import MetalPerformanceShadersGraph
import Foundation

let env = ProcessInfo.processInfo.environment
let args = Array(CommandLine.arguments.dropFirst())
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: stack_burst out_dir reference.NEF other.NEF ...\n".data(using: .utf8)!)
    exit(2)
}
let outDir = args[0], refPath = args[1], others = Array(args.dropFirst(2))
let mtl = MTLCreateSystemDefaultDevice()!
let lin = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CIContext(mtlDevice: mtl, options: [.workingColorSpace: lin, .cacheIntermediates: false])
let queue = mtl.makeCommandQueue()!
let R = Int(env["SEARCH"] ?? "") ?? 4
let T = 16, G = 8
let swapXY = env["SWAP_XY"] != nil
let rejectK = Double(env["REJECT"] ?? "") ?? 3
let useRaw8 = env["DECODER"] == "8"
var clock = Date()
func lap(_ label: String) { FileHandle.standardError.write(String(format: "%@ %.2fs\n", label, Date().timeIntervalSince(clock)).data(using: .utf8)!); clock = Date() }

func develop(_ path: String, nr: Bool) -> CIImage {
    let r = CIRAWFilter(imageURL: URL(fileURLWithPath: path))!
    r.decoderVersion = !useRaw8 && r.supportedDecoderVersions.contains(.version9) ? .version9 : .version8
    if r.isHighlightRecoverySupported { r.isHighlightRecoveryEnabled = true }
    if !nr {
        if r.isLuminanceNoiseReductionSupported { r.luminanceNoiseReductionAmount = 0 }
        if r.isColorNoiseReductionSupported { r.colorNoiseReductionAmount = 0 }
        if r.isSharpnessSupported { r.sharpnessAmount = 0 }
    }
    return r.outputImage!
}

let refImg = develop(refPath, nr: false)
let extent = refImg.extent
let W = Int(extent.width), H = Int(extent.height)
let Wp = (W + 63) / 64 * 64, Hg = (H + 1) / 2 * 2
let Wh = Wp / 2, Hh = Hg / 2

func render(_ img: CIImage) -> (MTLBuffer, CIImage) {
    let buf = mtl.makeBuffer(length: Wp * Hg * 8, options: .storageModeShared)!
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: W, height: H, mipmapped: false)
    td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .shared
    let tex = buf.makeTexture(descriptor: td, offset: 0, bytesPerRow: Wp * 8)!
    let dest = CIRenderDestination(mtlTexture: tex, commandBuffer: nil)
    dest.colorSpace = lin
    _ = try? ctx.startTask(toRender: img.cropped(to: extent).transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)), from: CGRect(x: 0, y: 0, width: W, height: H), to: dest, at: .zero).waitUntilCompleted()
    let view = CIImage(mtlTexture: tex, options: [.colorSpace: lin])!.transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
    return (buf, view)
}

func coarseShift(_ img: CIImage, against refView: CIImage) throws -> CGAffineTransform {
    let q: CGFloat = 0.25, small = CGAffineTransform(scaleX: q, y: q)
    let req = VNTranslationalImageRegistrationRequest(targetedCIImage: img.transformed(by: small))
    try VNImageRequestHandler(ciImage: refView.transformed(by: small)).perform([req])
    let t = (req.results?.first as? VNImageTranslationAlignmentObservation)?.alignmentTransform ?? .identity
    return CGAffineTransform(translationX: (t.tx / q).rounded(), y: (t.ty / q).rounded())
}

let g = MPSGraph()
let f32 = MPSDataType.float32
func c(_ v: Double) -> MPSGraphTensor { g.constant(v, dataType: f32) }
func n(_ v: [Int]) -> [NSNumber] { v.map { NSNumber(value: $0) } }
func pool(_ x: MPSGraphTensor, _ k: Int, _ s: Int, same: Bool = false) -> MPSGraphTensor {
    let d = MPSGraphPooling2DOpDescriptor(kernelWidth: k, kernelHeight: k, strideInX: s, strideInY: s, paddingStyle: same ? .TF_SAME : .TF_VALID, dataLayout: .NHWC)!
    return g.avgPooling2D(withSourceTensor: x, descriptor: d, name: nil)
}
let lumaWeights = g.constant([Float(0.2126), 0.7152, 0.0722, 0].withUnsafeBufferPointer { Data(buffer: $0) }, shape: n([1, 1, 1, 4]), dataType: f32)
func halfG(_ x: MPSGraphTensor) -> MPSGraphTensor {
    let y = g.reductionSum(with: g.multiplication(x, lumaWeights, name: nil), axis: 3, name: nil)
    return g.squareRoot(with: g.maximum(pool(y, 2, 2), c(0), name: nil), name: nil)
}
func conv3(_ x: MPSGraphTensor, _ k: [Float]) -> MPSGraphTensor {
    let w = g.constant(k.withUnsafeBufferPointer { Data(buffer: $0) }, shape: n([3, 3, 1, 1]), dataType: f32)
    let d = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingStyle: .TF_SAME, dataLayout: .NHWC, weightsLayout: .HWIO)!
    return g.convolution2D(x, weights: w, descriptor: d, name: nil)
}
func sum2(_ x: MPSGraphTensor) -> MPSGraphTensor { g.reductionSum(with: x, axes: n([3, 4]), name: nil) }

let refIn = g.placeholder(shape: n([1, Hg, Wp, 4]), dataType: .float16, name: "ref")
let frmIn = g.placeholder(shape: n([1, Hg, Wp, 4]), dataType: .float16, name: "frame")
let accIn = g.placeholder(shape: n([1, Hg, Wp, 4]), dataType: f32, name: "acc")
let wIn = g.placeholder(shape: n([1, Hg, Wp, 1]), dataType: f32, name: "wsum")
let ref = g.cast(refIn, to: f32, name: nil), frm = g.cast(frmIn, to: f32, name: nil)
let refG = halfG(ref), frmG = halfG(frm)
let lapK: [Float] = [1, -2, 1, -2, 4, -2, 1, -2, 1]
let sigma = g.multiplication(g.mean(of: g.absolute(with: conv3(refG, lapK), name: nil), axes: n([0, 1, 2, 3]), name: nil), c(sqrt(Double.pi / 2) / 6), name: nil)
let padded = g.padTensor(frmG, with: .clampToEdge, leftPadding: n([0, R, R, 0]), rightPadding: n([0, R, R, 0]), constantValue: 0, name: nil)
let K = 2 * R + 1
var costs: [MPSGraphTensor] = []
for oy in -R...R { for ox in -R...R {
    let s = g.sliceTensor(g.sliceTensor(padded, dimension: 1, start: R + oy, length: Hh, name: nil), dimension: 2, start: R + ox, length: Wh, name: nil)
    costs.append(pool(g.square(with: g.subtraction(s, refG, name: nil), name: nil), T, G))
} }
let gh = (Hh - T) / G + 1, gw = (Wh - T) / G + 1
let cost = g.concatTensors(costs, dimension: 3, name: nil)
let idx = g.reshape(g.reductionArgMinimum(with: cost, axis: 3, name: nil), shape: n([1, gh, gw]), name: nil)
let hot = g.reshape(g.oneHot(withIndicesTensor: idx, depth: K * K, axis: 3, dataType: f32, onValue: 1, offValue: 0, name: nil), shape: n([1, gh, gw, K, K]), name: nil)
let cube = g.reshape(cost, shape: n([1, gh, gw, K, K]), name: nil)
let c0 = sum2(g.multiplication(hot, cube, name: nil))
func subPixel(axis: Int) -> MPSGraphTensor {
    var l = [0, 0, 0, 0, 0], r = [0, 0, 0, 0, 0]; l[axis] = 1; r[axis] = 1
    let p = g.padTensor(cube, with: .constant, leftPadding: n(l), rightPadding: n(r), constantValue: 1e3, name: nil)
    let minus = sum2(g.multiplication(hot, g.sliceTensor(p, dimension: axis, start: 0, length: K, name: nil), name: nil))
    let plus = sum2(g.multiplication(hot, g.sliceTensor(p, dimension: axis, start: 2, length: K, name: nil), name: nil))
    let den = g.addition(g.subtraction(minus, g.multiplication(c0, c(2), name: nil), name: nil), plus, name: nil)
    let off = g.division(g.multiplication(g.subtraction(minus, plus, name: nil), c(0.5), name: nil), g.maximum(den, c(1e-12), name: nil), name: nil)
    return g.clamp(g.select(predicate: g.greaterThan(den, c(1e-12), name: nil), trueTensor: off, falseTensor: c(0), name: nil), min: c(-0.5), max: c(0.5), name: nil)
}
let idxF = g.cast(idx, to: f32, name: nil)
let iy = g.floor(with: g.division(idxF, c(Double(K)), name: nil), name: nil)
let ix = g.subtraction(idxF, g.multiplication(iy, c(Double(K)), name: nil), name: nil)
let fy = g.multiplication(g.addition(g.subtraction(g.reshape(iy, shape: n([1, gh, gw, 1]), name: nil), c(Double(R)), name: nil), g.reshape(subPixel(axis: 3), shape: n([1, gh, gw, 1]), name: nil), name: nil), c(2), name: nil)
let fx = g.multiplication(g.addition(g.subtraction(g.reshape(ix, shape: n([1, gh, gw, 1]), name: nil), c(Double(R)), name: nil), g.reshape(subPixel(axis: 4), shape: n([1, gh, gw, 1]), name: nil), name: nil), c(2), name: nil)
let binomial: [Float] = [1, 2, 1, 2, 4, 2, 1, 2, 1].map { $0 / 16 }
let edgeNorm = conv3(g.constant(1, shape: n([1, gh, gw, 1]), dataType: f32), binomial)
func smooth(_ x: MPSGraphTensor) -> MPSGraphTensor { g.division(conv3(x, binomial), edgeNorm, name: nil) }
let fieldGrid = g.concatTensors(swapXY ? [smooth(fy), smooth(fx)] : [smooth(fx), smooth(fy)], dimension: 3, name: nil)
let field = g.resize(fieldGrid, size: n([Hg, Wp]), mode: .bilinear, centerResult: true, alignCorners: false, layout: .NHWC, name: nil)
let warped = g.sampleGrid(withSourceTensor: frm, coordinateTensor: field, layout: .NHWC, normalizeCoordinates: false, relativeCoordinates: true, alignCorners: false, paddingMode: .clampToEdge, samplingMode: .bilinear, constantValue: 0, name: nil)
func centre(_ x: MPSGraphTensor) -> MPSGraphTensor {
    g.mean(of: g.sliceTensor(g.sliceTensor(x, dimension: 1, start: Hg / 4, length: Hg / 2, name: nil), dimension: 2, start: W / 4, length: W / 2, name: nil), axes: n([1, 2]), name: nil)
}
let gain = g.division(centre(ref), g.maximum(centre(warped), c(1e-6), name: nil), name: nil)
let aligned = g.multiplication(warped, gain, name: nil)
let diff = pool(pool(g.subtraction(halfG(aligned), refG, name: nil), 5, 1, same: true), 5, 1, same: true)
let thr = g.multiplication(sigma, c(rejectK * 0.192), name: nil)
let wHalf = g.exponent(with: g.negative(with: g.square(with: g.division(diff, thr, name: nil), name: nil), name: nil), name: nil)
let wFull = g.resize(wHalf, size: n([Hg, Wp]), mode: .bilinear, centerResult: true, alignCorners: false, layout: .NHWC, name: nil)
let accOut = g.addition(accIn, g.multiplication(aligned, wFull, name: nil), name: nil)
let wOut = g.addition(wIn, wFull, name: nil)
let meanW = g.mean(of: wHalf, axes: n([0, 1, 2, 3]), name: nil)
let maxShift = g.reductionMaximum(with: g.absolute(with: fieldGrid, name: nil), axes: n([0, 1, 2, 3]), name: nil)

let gi = MPSGraph()
let refOnly = gi.placeholder(shape: n([1, Hg, Wp, 4]), dataType: .float16, name: nil)
let accInit = gi.cast(refOnly, to: f32, name: nil)
let wInit = gi.constant(1, shape: n([1, Hg, Wp, 1]), dataType: f32)
let gf = MPSGraph()
let accFin = gf.placeholder(shape: n([1, Hg, Wp, 4]), dataType: f32, name: nil)
let wFin = gf.placeholder(shape: n([1, Hg, Wp, 1]), dataType: f32, name: nil)
let merged = gf.division(accFin, wFin, name: nil)
let coverage = gf.mean(of: wFin, axes: n([0, 1, 2, 3]), name: nil)

func tensor(_ b: MTLBuffer) -> MPSGraphTensorData { MPSGraphTensorData(b, shape: n([1, Hg, Wp, 4]), dataType: .float16) }
func scalar(_ t: MPSGraphTensorData) -> Float { var v: Float = 0; t.mpsndarray().readBytes(&v, strideBytes: nil); return v }

lap("setup")
let (refBuf, refView) = render(refImg)
let refData = tensor(refBuf)
let start = gi.run(with: queue, feeds: [refOnly: refData], targetTensors: [accInit, wInit], targetOperations: nil)
var acc = start[accInit]!, wsum = start[wInit]!
lap("reference decoded")
var frameReport: [[String: Any]] = []
for path in others {
    let (_, view) = render(develop(path, nr: false))
    let pre = try coarseShift(view, against: refView)
    let (buf, _) = render(view.transformed(by: pre))
    let r = g.run(with: queue, feeds: [refIn: refData, frmIn: tensor(buf), accIn: acc, wIn: wsum], targetTensors: [accOut, wOut, meanW, maxShift, sigma], targetOperations: nil)
    acc = r[accOut]!; wsum = r[wOut]!
    frameReport.append(["file": path, "kept": scalar(r[meanW]!), "global_shift": [pre.tx, pre.ty]])
    lap(String(format: "%@ global %.0f,%.0f  local max %.1f px  kept %.2f  noise %.4f", (path as NSString).lastPathComponent, pre.tx, pre.ty, scalar(r[maxShift]!), scalar(r[meanW]!), scalar(r[sigma]!)))
}
let fin = gf.run(with: queue, feeds: [accFin: acc, wFin: wsum], targetTensors: [merged, coverage], targetOperations: nil)
lap(String(format: "merged, mean frames per pixel %.2f", scalar(fin[coverage]!) ))
var px = [Float](repeating: 0, count: Hg * Wp * 4)
fin[merged]!.mpsndarray().readBytes(&px, strideBytes: nil)
let flip = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -CGFloat(Hg))
let stacked = CIImage(bitmapData: px.withUnsafeBufferPointer { Data(buffer: $0) }, bytesPerRow: Wp * 16, size: CGSize(width: Wp, height: Hg), format: .RGBAf, colorSpace: lin)
    .transformed(by: flip).cropped(to: CGRect(x: 0, y: Hg - H, width: W, height: H))
    .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY - CGFloat(Hg - H)))
    .settingAlphaOne(in: extent)
let name = ((refPath as NSString).lastPathComponent as NSString).deletingPathExtension
let summary: [String: Any] = ["reference": refPath, "frames": frameReport, "frames_per_pixel": scalar(fin[coverage]!), "width": W, "height": H]
try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted]).write(to: URL(fileURLWithPath: "\(outDir)/\(name)_stack.json"))
if env["RAW_OUT"] != nil {
    let rb = W * 8
    let half = NSMutableData(length: rb * H)!
    ctx.render(stacked, toBitmap: half.mutableBytes, rowBytes: rb, bounds: extent, format: .RGBAh, colorSpace: lin)
    try (half as Data).write(to: URL(fileURLWithPath: "\(outDir)/\(name)_stack.rgbah"))
}
if env["NO_TIFF"] == nil {
try ctx.writeTIFFRepresentation(of: stacked, to: URL(fileURLWithPath: "\(outDir)/\(name)_stack.tif"), format: .RGBA16, colorSpace: srgb)
}
lap("written")
if let crop = env["CROP"] {
    let v = crop.split(separator: ",").compactMap { Double($0) }
    let r = CGRect(x: v[0], y: v[1], width: v[2], height: v[2])
    let single = develop(refPath, nr: true)
    for (label, img) in [("stack", stacked), ("single_nr_off", refView), ("single_default", single)] {
        var out = [Float](repeating: 0, count: Int(v[2] * v[2]) * 4)
        ctx.render(img, toBitmap: &out, rowBytes: Int(v[2]) * 16, bounds: r, format: .RGBAf, colorSpace: lin)
        try Data(buffer: out.withUnsafeBufferPointer { $0 }).write(to: URL(fileURLWithPath: "\(outDir)/\(name)_\(label).rgba32f"))
    }
}
