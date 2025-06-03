//
//  minimal_test.mm
//  Minimal CoreML prediction test
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: ./minimal_test <model_path>\n");
            return 1;
        }
        
        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        
        printf("🧪 Minimal CoreML test\n");
        printf("📁 Model: %s\n", modelPath.UTF8String);
        
        // Load model (CPU only)
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUOnly;
        
        NSError *error = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
        
        if (!model) {
            printf("❌ Failed to load: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        
        printf("✅ Model loaded\n");
        
        // STEP 1: Create a new state object for stateful models
        printf("🔧 Creating new state for stateful model...\n");
        MLState *modelState = [model newState];
        if (!modelState) {
            printf("❌ Failed to create model state\n");
            return 1;
        }
        printf("✅ Model state created\n");
        
        // STEP 2: Create state feature values using the MLState object
        printf("🔧 Creating state feature values from MLState...\n");
        
        // The key insight: featureValueWithState: needs an MLState, not MLMultiArray!
        MLFeatureValue *keyCacheState = [MLFeatureValue featureValueWithState:modelState];
        MLFeatureValue *valueCacheState = [MLFeatureValue featureValueWithState:modelState];
        
        printf("✅ State feature values created\n");
        
        // STEP 3: Create regular inputs
        NSError *inputError = nil;
        
        // inputIds: [1, 1] with value 1 (minimal valid token)
        MLMultiArray *inputIds = [[MLMultiArray alloc] initWithShape:@[@(1), @(1)] 
                                                            dataType:MLMultiArrayDataTypeInt32 
                                                               error:&inputError];
        if (!inputIds) {
            printf("❌ Failed to create inputIds: %s\n", inputError.localizedDescription.UTF8String);
            return 1;
        }
        inputIds[@[@(0), @(0)]] = @(1); // Token ID 1
        
        // causalMask: [1, 1, 1, 1] with value 1.0
        MLMultiArray *causalMask = [[MLMultiArray alloc] initWithShape:@[@(1), @(1), @(1), @(1)] 
                                                              dataType:MLMultiArrayDataTypeFloat16 
                                                                 error:&inputError];
        if (!causalMask) {
            printf("❌ Failed to create causalMask: %s\n", inputError.localizedDescription.UTF8String);
            return 1;
        }
        causalMask[@[@(0), @(0), @(0), @(0)]] = @(1.0);
        
        // Create feature provider
        MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] 
            initWithDictionary:@{
                @"inputIds": [MLFeatureValue featureValueWithMultiArray:inputIds],
                @"causalMask": [MLFeatureValue featureValueWithMultiArray:causalMask]
            } error:&inputError];
        
        if (!input) {
            printf("❌ Failed to create input provider: %s\n", (providerError ?: inputError).localizedDescription.UTF8String);
            return 1;
        }
        
        printf("✅ Input prepared with states\n");
        printf("📊 inputIds: [1,1] = %d\n", [inputIds[@[@(0), @(0)]] intValue]);
        printf("📊 causalMask: [1,1,1,1] = %.1f\n", [causalMask[@[@(0), @(0), @(0), @(0)]] floatValue]);
        printf("📊 keyCache shape: %s\n", [[keyCache.shape description] UTF8String]);
        printf("📊 valueCache shape: %s\n", [[valueCache.shape description] UTF8String]);
        printf("📊 inputIds: [1,1] = %d\n", [inputIds[@[@(0), @(0)]] intValue]);
        printf("📊 causalMask: [1,1,1,1] = %.1f\n", [causalMask[@[@(0), @(0), @(0), @(0)]] floatValue]);
        
        // Test prediction with timeout simulation
        printf("🚀 Starting prediction...\n");
        NSDate *start = [NSDate date];
        
        // Set a shorter timeout and try the prediction
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        __block id<MLFeatureProvider> result = nil;
        __block NSError *predError = nil;
        
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
            // Try the actual prediction call
            MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
            result = [model predictionFromFeatures:input options:options error:&predError];
            dispatch_semaphore_signal(semaphore);
        });
        
        // Wait with 30 second timeout
        dispatch_time_t timeout = dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC);
        long waitResult = dispatch_semaphore_wait(semaphore, timeout);
        
        NSTimeInterval elapsed = [[NSDate date] timeIntervalSinceDate:start];
        
        if (waitResult == 0) {
            // Completed within timeout
            if (result && !predError) {
                printf("✅ Prediction successful in %.1f seconds!\n", elapsed);
                printf("📋 Output features: %s\n", [[[result featureNames] description] UTF8String]);
                
                // Try to get logits
                MLFeatureValue *logits = [result featureValueForName:@"logits"];
                if (logits) {
                    printf("📊 Logits type: %ld\n", (long)logits.type);
                    if (logits.type == MLFeatureTypeMultiArray) {
                        printf("📊 Logits shape: %s\n", [[logits.multiArrayValue.shape description] UTF8String]);
                    }
                }
                
            } else {
                printf("❌ Prediction failed: %s\n", predError.localizedDescription.UTF8String);
                return 1;
            }
        } else {
            printf("⏰ Prediction timed out after 30 seconds\n");
            printf("🔍 This confirms the CoreML prediction call is hanging\n");
            return 1;
        }
        
        printf("🎉 Test completed successfully!\n");
    }
    return 0;
}
