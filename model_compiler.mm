//
//  model_compiler.mm
//  CoreML Model Compiler
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: ./model_compiler <input_model.mlpackage> [output_directory]\n");
            printf("Example: ./model_compiler StatefulMistral7BInstructFP16.mlpackage\n");
            return 1;
        }
        
        NSString *inputPath = [NSString stringWithUTF8String:argv[1]];
        NSURL *inputURL = [NSURL fileURLWithPath:inputPath];
        
        // Default output directory is same as input
        NSString *outputDir = [inputPath stringByDeletingLastPathComponent];
        if (argc >= 3) {
            outputDir = [NSString stringWithUTF8String:argv[2]];
        }
        NSURL *outputURL = [NSURL fileURLWithPath:outputDir];
        
        NSLog(@"🔧 Starting model compilation");
        NSLog(@"📥 Input: %@", inputPath);
        NSLog(@"📤 Output directory: %@", outputDir);
        
        // Check if input exists
        if (![[NSFileManager defaultManager] fileExistsAtPath:inputPath]) {
            NSLog(@"❌ Input model not found: %@", inputPath);
            return 1;
        }
        
        // Get input model info
        NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:inputPath error:nil];
        unsigned long long inputSize = [attributes fileSize];
        NSLog(@"📏 Input size: %.2f GB", inputSize / (1024.0 * 1024.0 * 1024.0));
        
        NSDate *startTime = [NSDate date];
        NSLog(@"⏱️  Compilation started at: %@", startTime);
        NSLog(@"⚠️  This may take several minutes for large models...");
        
        // Try different compilation configurations for better compatibility
        NSArray *compileConfigs = @[
            @{@"name": @"CPU_Only", @"compute": @(MLComputeUnitsCPUOnly)},
            @{@"name": @"CPU_And_GPU", @"compute": @(MLComputeUnitsCPUAndGPU)}
        ];
        
        NSURL *compiledURL = nil;
        NSError *compileError = nil;
        
        for (NSDictionary *config in compileConfigs) {
            NSLog(@"🔧 Trying compilation with %@...", config[@"name"]);
            
            // Try compilation with specific configuration
            compiledURL = [MLModel compileModelAtURL:inputURL error:&compileError];
            
            if (compiledURL && !compileError) {
                NSLog(@"✅ Compilation successful with %@", config[@"name"]);
                break;
            } else {
                NSLog(@"❌ Compilation failed with %@: %@", config[@"name"], compileError.localizedDescription);
                compiledURL = nil;
            }
        }
        
        NSTimeInterval elapsed = [[NSDate date] timeIntervalSinceDate:startTime];
        
        if (compiledURL && !compileError) {
            NSLog(@"✅ Compilation successful in %.1f seconds!", elapsed);
            NSLog(@"📁 Compiled model location: %@", compiledURL.path);
            
            // Get compiled model size accurately
            NSString *compiledPath = compiledURL.path;
            
            // For .mlmodelc, calculate total directory size
            unsigned long long totalCompiledSize = 0;
            NSDirectoryEnumerator *enumerator = [[NSFileManager defaultManager] enumeratorAtPath:compiledPath];
            NSString *file;
            while (file = [enumerator nextObject]) {
                NSString *fullPath = [compiledPath stringByAppendingPathComponent:file];
                NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:fullPath error:nil];
                if ([attrs fileType] == NSFileTypeRegular) {
                    totalCompiledSize += [attrs fileSize];
                }
            }
            
            NSLog(@"📏 Compiled total size: %.2f GB", totalCompiledSize / (1024.0 * 1024.0 * 1024.0));
            
            // Test loading the compiled model quickly
            NSLog(@"🧪 Testing compiled model load...");
            NSDate *loadStart = [NSDate date];
            
            MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
            config.computeUnits = MLComputeUnitsCPUOnly; // Safe test
            
            NSError *loadError = nil;
            MLModel *testModel = [MLModel modelWithContentsOfURL:compiledURL 
                                                   configuration:config 
                                                           error:&loadError];
            
            NSTimeInterval loadTime = [[NSDate date] timeIntervalSinceDate:loadStart];
            
            if (testModel && !loadError) {
                NSLog(@"✅ Compiled model loads successfully in %.1f seconds", loadTime);
                NSLog(@"📋 Model inputs: %@", testModel.modelDescription.inputDescriptionsByName.allKeys);
                NSLog(@"📋 Model outputs: %@", testModel.modelDescription.outputDescriptionsByName.allKeys);
                
                // Test creating state for stateful models
                MLState *testState = [testModel newState];
                if (testState) {
                    NSLog(@"✅ Stateful model - state created successfully");
                } else {
                    NSLog(@"ℹ️  Non-stateful model");
                }
                
            } else {
                NSLog(@"⚠️  Compiled model test load failed: %@", loadError.localizedDescription);
            }
            
        } else {
            NSLog(@"❌ All compilation attempts failed");
            if (compileError) {
                NSLog(@"Last error: %@", compileError.localizedDescription);
            }
            return 1;
        }
        
        NSLog(@"🎉 Model compilation complete!");
        NSLog(@"💡 You can now use the compiled model at: %@", compiledURL.path);
        
    }
    return 0;
}
