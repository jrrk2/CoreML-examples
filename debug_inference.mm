//
//  debug_inference.mm
//  Debug inference step by step
//

#import <Foundation/Foundation.h>
#import "LlamaInference.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            printf("Usage: ./debug_inference <model_path> <prompt>\n");
            return 1;
        }
        
        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        NSString *prompt = [NSString stringWithUTF8String:argv[2]];
        
        printf("🔍 Debug inference step by step\n");
        printf("📁 Model: %s\n", modelPath.UTF8String);
        printf("💬 Prompt: %s\n", prompt.UTF8String);
        
        LlamaInference *inference = [[LlamaInference alloc] initWithModelPath:modelPath];
        
        // Step 1: Load model with timeout
        printf("\n🔄 Step 1: Loading model...\n");
        dispatch_semaphore_t loadSemaphore = dispatch_semaphore_create(0);
        __block BOOL loadSuccess = NO;
        
        [inference loadModelWithCompletion:^(BOOL success, NSError *error) {
            loadSuccess = success;
            if (!success) {
                printf("❌ Load failed: %s\n", error.localizedDescription.UTF8String);
            } else {
                printf("✅ Model loaded successfully\n");
            }
            dispatch_semaphore_signal(loadSemaphore);
        }];
        
        // Wait for load with timeout
        dispatch_time_t loadTimeout = dispatch_time(DISPATCH_TIME_NOW, 120 * NSEC_PER_SEC);
        if (dispatch_semaphore_wait(loadSemaphore, loadTimeout) != 0) {
            printf("⏰ Model loading timed out after 2 minutes\n");
            return 1;
        }
        
        if (!loadSuccess) {
            return 1;
        }
        
        // Step 2: Test input preparation
        printf("\n🔄 Step 2: Testing input preparation...\n");
        NSError *error = nil;
        
        // Create a test instance to access private methods
        // We'll need to expose prepareInputWithText for testing
        printf("⚠️  Input preparation test needs method exposure\n");
        
        // Step 3: Test inference with timeout
        printf("\n🔄 Step 3: Testing inference with timeout...\n");
        dispatch_semaphore_t inferSemaphore = dispatch_semaphore_create(0);
        __block NSString *result = nil;
        __block NSError *inferError = nil;
        
        NSDate *inferStart = [NSDate date];
        printf("⏱️  Inference started at: %s\n", [[inferStart description] UTF8String]);
        
        [inference generateTextWithPrompt:prompt 
                               completion:^(NSString *generatedText, NSError *error) {
            result = generatedText;
            inferError = error;
            
            NSTimeInterval elapsed = [[NSDate date] timeIntervalSinceDate:inferStart];
            printf("⏱️  Inference completed in %.1f seconds\n", elapsed);
            
            if (error) {
                printf("❌ Inference error: %s\n", error.localizedDescription.UTF8String);
            } else {
                printf("✅ Inference successful\n");
                printf("📝 Result: %s\n", generatedText.UTF8String);
            }
            dispatch_semaphore_signal(inferSemaphore);
        }];
        
        // Wait for inference with timeout (2 minutes)
        dispatch_time_t inferTimeout = dispatch_time(DISPATCH_TIME_NOW, 120 * NSEC_PER_SEC);
        long inferResult = dispatch_semaphore_wait(inferSemaphore, inferTimeout);
        
        if (inferResult != 0) {
            printf("⏰ Inference timed out after 2 minutes\n");
            printf("🔍 Process likely stuck in model prediction\n");
            return 1;
        }
        
        printf("\n🏁 Debug complete\n");
    }
    return 0;
}
