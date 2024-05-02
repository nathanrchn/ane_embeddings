import CoreML
import Foundation

extension MLMultiArray {
    class func initIOSurfaceArray(shape: [NSNumber], value: Float16) -> MLMultiArray? {
        guard let width = shape.last?.intValue else { return nil }
        let height = shape[0..<shape.count-1].reduce(1, { $0 * $1.intValue })

        var pixelBuffer: CVPixelBuffer?
        let createReturn = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_OneComponent16Half,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary,
            &pixelBuffer)
        guard createReturn == kCVReturnSuccess else { return nil }
        guard let pixelBuffer = pixelBuffer else { return nil }

        let mlMultiArray = MLMultiArray(pixelBuffer: pixelBuffer, shape: shape)
        
        CVPixelBufferLockBaseAddress(mlMultiArray.pixelBuffer!, CVPixelBufferLockFlags.init(rawValue: 0))
        
        let pointer = CVPixelBufferGetBaseAddress(mlMultiArray.pixelBuffer!)!
        pointer.initializeMemory(as: Float16.self, repeating: value, count: mlMultiArray.count)
        
        CVPixelBufferUnlockBaseAddress(mlMultiArray.pixelBuffer!, CVPixelBufferLockFlags.init(rawValue: 0))
        return mlMultiArray
    }
}

let configuration = MLModelConfiguration()
configuration.computeUnits = MLComputeUnits.cpuAndNeuralEngine

let model = try! AneEmbeddings(configuration: configuration)

let input_ids = MLMultiArray.initIOSurfaceArray(shape: [8, 384], value: 1)!

let n = 10

print("running model")
let start = Date()
for _ in 0..<n {
    let _ = try model.prediction(input_ids: input_ids)
}
let end = Date()

print("time in ms: ", Double(end.timeIntervalSince(start)) * 1000 / Double(n))
print("tokens per sec: ", Double(n) / Double(end.timeIntervalSince(start)))
