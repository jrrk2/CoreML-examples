//
//  debug_loader.mm
//  Minimal CoreML loader for debugging
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <mach/mach.h>

void logMemoryUsage() {
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    kern_return_t kerr = task_info(mach_task_self(), TASK_BASIC_INFO, 
                                   (task_info_t)&info, &size);
    if (kerr == KERN_SUCCESS) {
        float memGB = info.resident_size / (1024.0 * 1024.0 * 1024.0);
        NSLog(@"📊 Current memory usage: %.2f GB", memGB);
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: ./debug_loader <model_path>\n");
            return 1;
        }
        
        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        
        NSLog(@"🔍 Starting debug loader");
        NSLog(@"📁 Model path: %@", modelPath);
        
        // Check if file exists
        if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
            NSLog(@"❌ Model file not found");
            return 1;
        }
        
        // Get file size
        NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:modelPath error:nil];
        unsigned long long fileSize = [attributes fileSize];
        NSLog(@"📏 Model package size: %.2f GB", fileSize / (1024.0 * 1024.0 * 1024.0));
        
        // Check weight.bin size specifically
        NSString *weightPath = [modelPath stringByAppendingPathComponent:@"Data/com.apple.CoreML/weights/weight.bin"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:weightPath]) {
            NSDictionary *weightAttrs = [[NSFileManager defaultManager] attributesOfItemAtPath:weightPath error:nil];
            unsigned long long weightSize = [weightAttrs fileSize];
            NSLog(@"⚖️  Weight file size: %.2f GB", weightSize / (1024.0 * 1024.0 * 1024.0));
        }
        
        logMemoryUsage();
        
        // Try different configurations
        NSArray *testConfigs = @[
            @{@"name": @"CPU_Only", @"compute": @(MLComputeUnitsCPUOnly)},
            @{@"name": @"CPU_And_GPU", @"compute": @(MLComputeUnitsCPUAndGPU)},
            @{@"name": @"All", @"compute": @(MLComputeUnitsAll)}
        ];
        
        for (NSDictionary *config in testConfigs) {
            NSLog(@"\n🧪 Testing configuration: %@", config[@"name"]);
            
            MLModelConfiguration *mlConfig = [[MLModelConfiguration alloc] init];
            mlConfig.computeUnits = (MLComputeUnits)[config[@"compute"] integerValue];
            
            NSDate *startTime = [NSDate date];
            NSLog(@"⏱️  Load start: %@", startTime);
            
            // Set a reasonable timeout
            dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
            __block MLModel *model = nil;
            __block NSError *error = nil;
            __block BOOL completed = NO;
            
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
                model = [MLModel modelWithContentsOfURL:modelURL 
                                           configuration:mlConfig 
                                                   error:&error];
                completed = YES;
                dispatch_semaphore_signal(semaphore);
            });
            
            // Wait with timeout (3 minutes)
            dispatch_time_t timeout = dispatch_time(DISPATCH_TIME_NOW, 180 * NSEC_PER_SEC);
            long result = dispatch_semaphore_wait(semaphore, timeout);
            
            if (result == 0 && completed) {
                NSTimeInterval elapsed = [[NSDate date] timeIntervalSinceDate:startTime];
                if (model && !error) {
                    NSLog(@"✅ SUCCESS with %@ in %.1f seconds", config[@"name"], elapsed);
                    NSLog(@"📋 Model description: %@", model.modelDescription.inputDescriptionsByName.allKeys);
                    logMemoryUsage();
                    break; // Success, stop testing
                } else {
                    NSLog(@"❌ FAILED with %@: %@", config[@"name"], error.localizedDescription);
                }
            } else {
                NSLog(@"⏰ TIMEOUT with %@ after 3 minutes", config[@"name"]);
                // Kill the background task
            }
            
            logMemoryUsage();
        }
        
        NSLog(@"🏁 Debug complete");
    }
    return 0;
}
