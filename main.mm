//
//  main.mm
//  CoreML Llama 2 Demonstrator
//

#import <Foundation/Foundation.h>
#import "LlamaInference.h"

void printUsage() {
    printf("Usage: ./demo <path_to_model> [prompt]\n");
    printf("Example: ./demo ~/models/llama-2-7b-chat.mlpackage \"Hello, how are you?\"\n");
}

void runInteractiveMode(LlamaInference *inference) {
    printf("\n🚀 Interactive mode started. Type 'quit' to exit.\n\n");
    
    char buffer[1024];
    while (true) {
        printf("You: ");
        fflush(stdout);
        
        if (!fgets(buffer, sizeof(buffer), stdin)) {
            break;
        }
        
        NSString *input = [[NSString stringWithUTF8String:buffer] 
                          stringByTrimmingCharactersInSet:[NSCharacterSet newlineCharacterSet]];
        
        if ([input isEqualToString:@"quit"]) {
            break;
        }
        
        if (input.length == 0) {
            continue;
        }
        
        printf("Assistant: ");
        fflush(stdout);
        
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        
        [inference generateTextWithPrompt:input completion:^(NSString *result, NSError *error) {
            if (error) {
                printf("❌ Error: %s\n", error.localizedDescription.UTF8String);
            } else {
                printf("%s\n", result.UTF8String);
            }
            printf("\n");
            dispatch_semaphore_signal(semaphore);
        }];
        
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printUsage();
            return 1;
        }
        
        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        
        // Check if model path exists
        if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
            printf("❌ Model file not found: %s\n", modelPath.UTF8String);
            return 1;
        }
        
        printf("🔄 Loading model from: %s\n", modelPath.UTF8String);
        
        LlamaInference *inference = [[LlamaInference alloc] initWithModelPath:modelPath];
        
        dispatch_semaphore_t loadSemaphore = dispatch_semaphore_create(0);
        __block BOOL loadSuccess = NO;
        
        [inference loadModelWithCompletion:^(BOOL success, NSError *error) {
            loadSuccess = success;
            if (!success) {
                printf("❌ Failed to load model: %s\n", error.localizedDescription.UTF8String);
            }
            dispatch_semaphore_signal(loadSemaphore);
        }];
        
        dispatch_semaphore_wait(loadSemaphore, DISPATCH_TIME_FOREVER);
        
        if (!loadSuccess) {
            return 1;
        }
        
        if (argc >= 3) {
            // Single prompt mode
            NSString *prompt = [NSString stringWithUTF8String:argv[2]];
            printf("🤖 Generating response for: %s\n", prompt.UTF8String);
            
            dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
            
            [inference generateTextWithPrompt:prompt completion:^(NSString *result, NSError *error) {
                if (error) {
                    printf("❌ Error: %s\n", error.localizedDescription.UTF8String);
                } else {
                    printf("Response: %s\n", result.UTF8String);
                }
                dispatch_semaphore_signal(semaphore);
            }];
            
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        } else {
            // Interactive mode
            runInteractiveMode(inference);
        }
        
        printf("👋 Goodbye!\n");
    }
    
    return 0;
}
