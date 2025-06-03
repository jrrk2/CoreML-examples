//
//  stateful_test.mm
//  Clean stateful model test with proper compile-time checking
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

// Forward declare the method to avoid runtime surprises
@interface MLFeatureValue (StateExtensions)
+ (instancetype)featureValueWithState:(MLState *)state;
@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: ./stateful_test <model_path>\n");
            return 1;
        }
        
        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        
        printf("🧪 Clean stateful model test\n");
        printf("📁 Model: %s\n", modelPath.UTF8String);
        
        // Load model (CPU only for reliability)
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUOnly;
        
        NSError *error = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
        
        if (!model) {
            printf("❌ Failed to load: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        
        printf("✅ Model loaded\n");
        
        // Create new state
        MLState *modelState = [model newState];
        if (!modelState) {
            printf("❌ Failed to create model state\n");
            return 1;
        }
        printf("✅ Model state created: %s\n", [[modelState description] UTF8String]);
        
        // Create minimal inputs
        NSError *inputError = nil;
        
        MLMultiArray *inputIds = [[MLMultiArray alloc] initWithShape:@[@(1), @(1)] 
                                                            dataType:MLMultiArrayDataTypeInt32 
                                                               error:&inputError];
        inputIds[@[@(0), @(0)]] = @(1);
        
        MLMultiArray *causalMask = [[MLMultiArray alloc] initWithShape:@[@(1), @(1), @(1), @(1)] 
                                                              dataType:MLMultiArrayDataTypeFloat16 
                                                                 error:&inputError];
        causalMask[@[@(0), @(0), @(0), @(0)]] = @(1.0);
        
        // Test 1: Try without explicit state inputs (just basic inputs)
        printf("\n🧪 Test 1: Basic inputs only\n");
        MLDictionaryFeatureProvider *basicInput = [[MLDictionaryFeatureProvider alloc] 
            initWithDictionary:@{
                @"inputIds": [MLFeatureValue featureValueWithMultiArray:inputIds],
                @"causalMask": [MLFeatureValue featureValueWithMultiArray:causalMask]
            } error:&inputError];
        
        // Try stateful prediction
        NSDate *start1 = [NSDate date];
        id<MLFeatureProvider> result1 = [model predictionFromFeatures:basicInput 
                                                            usingState:modelState 
                                                                 error:&error];
        NSTimeInterval elapsed1 = [[NSDate date] timeIntervalSinceDate:start1];
        
        if (result1) {
            printf("✅ Test 1 SUCCESS in %.2f seconds\n", elapsed1);
            printf("📋 Output features: %s\n", [[[result1 featureNames] description] UTF8String]);
        } else {
            printf("❌ Test 1 FAILED: %s\n", error.localizedDescription.UTF8String);
            
            // Test 2: Try with explicit state inputs
            printf("\n🧪 Test 2: With explicit state inputs\n");
            
            // Check if the method exists at compile time
            if ([MLFeatureValue respondsToSelector:@selector(featureValueWithState:)]) {
                MLFeatureValue *stateFeature = [MLFeatureValue featureValueWithState:modelState];
                
                MLDictionaryFeatureProvider *fullInput = [[MLDictionaryFeatureProvider alloc] 
                    initWithDictionary:@{
                        @"inputIds": [MLFeatureValue featureValueWithMultiArray:inputIds],
                        @"causalMask": [MLFeatureValue featureValueWithMultiArray:causalMask],
                        @"keyCache": stateFeature,
                        @"valueCache": stateFeature
                    } error:&inputError];
                
                NSDate *start2 = [NSDate date];
                id<MLFeatureProvider> result2 = [model predictionFromFeatures:fullInput 
                                                                    usingState:modelState 
                                                                         error:&error];
                NSTimeInterval elapsed2 = [[NSDate date] timeIntervalSinceDate:start2];
                
                if (result2) {
                    printf("✅ Test 2 SUCCESS in %.2f seconds\n", elapsed2);
                    printf("📋 Output features: %s\n", [[[result2 featureNames] description] UTF8String]);
                } else {
                    printf("❌ Test 2 FAILED: %s\n", error.localizedDescription.UTF8String);
                }
            } else {
                printf("❌ featureValueWithState: method not available\n");
            }
        }
        
        printf("\n🏁 Stateful test complete\n");
    }
    return 0;
}
