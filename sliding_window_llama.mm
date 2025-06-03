//
//  sliding_window_llama.mm
//  Interactive LLaMA with sliding window context management
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <iostream>
#include <string>

// Global state
static NSDictionary *g_vocabulary = nil;
static NSDictionary *g_invVocabulary = nil;
static MLModel *g_model = nil;
static MLMultiArray *g_inputIds = nil;
static MLMultiArray *g_attentionMask = nil;
static MLDictionaryFeatureProvider *g_input = nil;
static NSInteger g_seqLength = 64;
static NSMutableArray *g_conversationTokens = nil; // Full conversation history

// Load vocabulary from tokenizer.json
BOOL loadVocabulary(NSString *modelPackagePath) {
    NSString *tokenizerPath = [modelPackagePath stringByAppendingPathComponent:@"tokenizer.json"];
    
    if (![[NSFileManager defaultManager] fileExistsAtPath:tokenizerPath]) {
        printf("⚠️  tokenizer.json not found\n");
        return NO;
    }
    
    NSError *error = nil;
    NSData *tokenizerData = [NSData dataWithContentsOfFile:tokenizerPath];
    NSDictionary *tokenizerJson = [NSJSONSerialization JSONObjectWithData:tokenizerData 
                                                                    options:0 
                                                                      error:&error];
    
    if (!tokenizerJson || error) {
        printf("❌ Failed to parse tokenizer.json: %s\n", error.localizedDescription.UTF8String);
        return NO;
    }
    
    NSDictionary *model = tokenizerJson[@"model"];
    g_vocabulary = model[@"vocab"];
    
    if (!g_vocabulary) {
        printf("❌ No vocab found in tokenizer.json\n");
        return NO;
    }
    
    // Create inverse mapping
    NSMutableDictionary *idToToken = [NSMutableDictionary dictionary];
    for (NSString *token in g_vocabulary) {
        NSNumber *tokenId = g_vocabulary[token];
        idToToken[tokenId] = token;
    }
    g_invVocabulary = [idToToken copy];
    
    printf("✅ Loaded vocabulary with %lu tokens\n", (unsigned long)g_vocabulary.count);
    return YES;
}

// SentencePiece tokenization
NSArray* tokenizeText(NSString *text) {
    NSMutableArray *tokens = [NSMutableArray array];
    
    // Add BOS token
    [tokens addObject:@(1)]; // <s>
    
    // Convert to SentencePiece format
    NSString *spText = [@"▁" stringByAppendingString:
                       [text stringByReplacingOccurrencesOfString:@" " withString:@"▁"]];
    
    // Greedy longest-match tokenization
    NSInteger pos = 0;
    while (pos < spText.length) {
        NSString *bestMatch = nil;
        NSNumber *bestTokenId = nil;
        NSInteger bestLength = 0;
        
        for (NSInteger len = MIN(20, spText.length - pos); len >= 1; len--) {
            NSString *substr = [spText substringWithRange:NSMakeRange(pos, len)];
            NSNumber *tokenId = g_vocabulary[substr];
            
            if (tokenId && len > bestLength) {
                bestMatch = substr;
                bestTokenId = tokenId;
                bestLength = len;
            }
        }
        
        if (bestMatch) {
            [tokens addObject:bestTokenId];
            pos += bestLength;
        } else {
            [tokens addObject:@(0)]; // <unk>
            pos += 1;
        }
    }
    
    return [tokens copy];
}

// Create chat format tokens
NSArray* createChatTokens(NSString *userInput) {
    NSString *chatFormat = [NSString stringWithFormat:@"[INST] %@ [/INST]", userInput];
    return tokenizeText(chatFormat);
}

// Detokenize token ID back to text
NSString* detokenize(NSInteger tokenId) {
    NSString *token = g_invVocabulary[@(tokenId)];
    if (token) {
        NSString *text = [token stringByReplacingOccurrencesOfString:@"▁" withString:@" "];
        
        // Handle special tokens - convert hex escape sequences to actual characters
        if ([text isEqualToString:@"<0x0A>"]) {
            return @"\n";  // Line feed (LF)
        }
        if ([text isEqualToString:@"<0x0D>"]) {
            return @"\r";  // Carriage return (CR)
        }
        
        return text;
    }
    return [NSString stringWithFormat:@"[UNK_%ld]", tokenId];
}

// Check if generation should stop
BOOL shouldStopGeneration(NSInteger tokenId) {
    return (tokenId == 2 ||      // </s> (EOS)
            tokenId == 0);       // <unk>
}

// Sliding window context management with improved conversation flow
NSArray* createSlidingWindow(NSArray *fullTokens, NSInteger maxLength) {
    if (fullTokens.count <= maxLength) {
        return fullTokens;
    }
    
    // Strategy: Preserve conversation flow and code context better
    NSInteger keepStructural = 8;    // Always keep BOS + initial INST structure
    NSInteger keepRecent = maxLength - keepStructural;  // Most space for recent context
    
    NSMutableArray *slidingWindow = [NSMutableArray array];
    
    // Always keep initial structural tokens (BOS, first INST markers)
    NSInteger structuralCount = MIN(keepStructural, fullTokens.count);
    for (NSInteger i = 0; i < structuralCount; i++) {
        [slidingWindow addObject:fullTokens[i]];
    }
    
    // Keep the most recent context (this preserves conversation flow)
    NSInteger startRecent = MAX(structuralCount, fullTokens.count - keepRecent);
    for (NSInteger i = startRecent; i < fullTokens.count; i++) {
        [slidingWindow addObject:fullTokens[i]];
    }
    
    // Only log when windowing actually happens
    static NSInteger lastLoggedSize = 0;
    if (fullTokens.count > lastLoggedSize + 20) { // Log every 20 tokens
        printf("🪟 Sliding window: %lu→%lu tokens\n", 
               (unsigned long)fullTokens.count, (unsigned long)slidingWindow.count);
        lastLoggedSize = fullTokens.count;
    }
    
    return [slidingWindow copy];
}

// Update input tensors with sliding window
void updateInputTensors(NSArray *tokens) {
    // Apply sliding window
    NSArray *windowTokens = createSlidingWindow(tokens, g_seqLength - 10); // Leave room for generation
    
    // Clear tensors
    for (NSInteger i = 0; i < g_seqLength; i++) {
        g_inputIds[@[@(0), @(i)]] = @(1); // Pad with <s>
        g_attentionMask[@[@(0), @(i)]] = @(0); // Don't attend to padding
    }
    
    // Fill with windowed tokens
    NSInteger tokenCount = MIN(windowTokens.count, g_seqLength);
    for (NSInteger i = 0; i < tokenCount; i++) {
        g_inputIds[@[@(0), @(i)]] = windowTokens[i];
        g_attentionMask[@[@(0), @(i)]] = @(1); // Attend to real tokens
    }
}

// Initialize model
BOOL initializeModel(NSString *modelPath) {
    printf("🔧 Initializing Sliding Window LLaMA Engine...\n");
    printf("📥 Model: %s\n", modelPath.UTF8String);
    
    if (!loadVocabulary(modelPath)) {
        return NO;
    }
    
    NSURL *inputURL = [NSURL fileURLWithPath:modelPath];
    
    // Compile and load model
    printf("⚙️  Compiling model...\n");
    NSError *compileError = nil;
    NSURL *compiledURL = [MLModel compileModelAtURL:inputURL error:&compileError];
    
    if (!compiledURL) {
        printf("❌ Compilation failed: %s\n", compileError.localizedDescription.UTF8String);
        return NO;
    }
    
    printf("🧠 Loading model...\n");
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsAll;
    
    NSError *loadError = nil;
    g_model = [MLModel modelWithContentsOfURL:compiledURL 
                                configuration:config 
                                        error:&loadError];
    
    if (!g_model && loadError) {
        printf("⚠️  Neural Engine failed, trying CPU...\n");
        config.computeUnits = MLComputeUnitsCPUOnly;
        g_model = [MLModel modelWithContentsOfURL:compiledURL 
                                    configuration:config 
                                            error:&loadError];
    }
    
    if (!g_model) {
        printf("❌ Model loading failed: %s\n", loadError.localizedDescription.UTF8String);
        return NO;
    }
    
    NSString *compute = (config.computeUnits == MLComputeUnitsAll) ? @"Neural Engine" : @"CPU";
    printf("✅ Model loaded using %s\n", compute.UTF8String);
    
    // Auto-detect sequence length
    NSDictionary *inputDesc = g_model.modelDescription.inputDescriptionsByName;
    MLFeatureDescription *inputIdsDesc = inputDesc[@"input_ids"];
    if (inputIdsDesc.multiArrayConstraint.shape.count >= 2) {
        g_seqLength = [inputIdsDesc.multiArrayConstraint.shape[1] integerValue];
    }
    printf("📐 Detected sequence length: %ld\n", g_seqLength);
    
    // Create reusable tensors
    NSError *inputError = nil;
    g_inputIds = [[MLMultiArray alloc] initWithShape:@[@(1), @(g_seqLength)] 
                                            dataType:MLMultiArrayDataTypeInt32 
                                               error:&inputError];
    
    g_attentionMask = [[MLMultiArray alloc] initWithShape:@[@(1), @(g_seqLength)] 
                                                 dataType:MLMultiArrayDataTypeInt32 
                                                    error:&inputError];
    
    if (!g_inputIds || !g_attentionMask) {
        printf("❌ Input tensor creation failed\n");
        return NO;
    }
    
    g_input = [[MLDictionaryFeatureProvider alloc] 
        initWithDictionary:@{
            @"input_ids": [MLFeatureValue featureValueWithMultiArray:g_inputIds],
            @"attention_mask": [MLFeatureValue featureValueWithMultiArray:g_attentionMask]
        } error:&inputError];
    
    if (!g_input) {
        printf("❌ Input provider creation failed\n");
        return NO;
    }
    
    // Initialize conversation history
    g_conversationTokens = [NSMutableArray array];
    
    printf("🎉 Initialization complete!\n");
    printf("🪟 Sliding window enabled for unlimited context\n");
    
    return YES;
}

// Generate response with improved context awareness
NSString* generateResponse(NSString *userInput) {
    // Check if this is a continuation request
    BOOL isContinuation = ([userInput isEqualToString:@"continue"] || 
                          [userInput isEqualToString:@"more"] ||
                          [userInput hasPrefix:@"continue"]);
    
    if (!isContinuation) {
        // Add new user input to conversation history
        NSArray *inputTokens = createChatTokens(userInput);
        [g_conversationTokens addObjectsFromArray:inputTokens];
        
        printf("🔤 Added %lu tokens to conversation (%lu total)\n", 
               (unsigned long)inputTokens.count, (unsigned long)g_conversationTokens.count);
    } else {
        printf("▶️  Continuing previous response...\n");
    }
    
    NSMutableString *response = [NSMutableString string];
    printf("🤖 Assistant: ");
    fflush(stdout);
    
    // Generate tokens with context-aware sliding window
    NSInteger maxGenerationSteps = 100;
    
    for (NSInteger step = 0; step < maxGenerationSteps; step++) {
        // Update tensors with current conversation using improved sliding window
        updateInputTensors(g_conversationTokens);
        
        // Run inference
        NSError *inferError = nil;
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        id<MLFeatureProvider> result = [g_model predictionFromFeatures:g_input 
                                                               options:options 
                                                                 error:&inferError];
        
        if (!result) {
            printf("\n❌ Inference failed: %s\n", inferError.localizedDescription.UTF8String);
            break;
        }
        
        // Extract logits and find best token
        MLFeatureValue *logits = [result featureValueForName:@"logits"];
        if (!logits || logits.type != MLFeatureTypeMultiArray) {
            printf("\n❌ Invalid logits\n");
            break;
        }
        
        MLMultiArray *logitsArray = logits.multiArrayValue;
        NSInteger vocabSize = [logitsArray.shape[2] integerValue];
        
        // Find current window and predict from its last position
        NSArray *currentWindow = createSlidingWindow(g_conversationTokens, g_seqLength - 10);
        NSInteger predictionPos = MIN(currentWindow.count - 1, g_seqLength - 1);
        
        // Greedy token selection
        float bestLogit = -INFINITY;
        NSInteger bestToken = 0;
        
        for (NSInteger i = 0; i < vocabSize; i++) {
            NSNumber *logitVal = logitsArray[@[@(0), @(predictionPos), @(i)]];
            float logit = [logitVal floatValue];
            
            if (logit > bestLogit) {
                bestLogit = logit;
                bestToken = i;
            }
        }
        
        // Check for stop condition
        if (shouldStopGeneration(bestToken)) {
            printf("\n🛑 EOS detected\n");
            break;
        }
        
        // Add token to conversation and display
        [g_conversationTokens addObject:@(bestToken)];
        NSString *tokenText = detokenize(bestToken);
        [response appendString:tokenText];
        printf("%s", tokenText.UTF8String);
        fflush(stdout);
        
        // Improved stopping logic - be more careful about when to stop
        if ([tokenText containsString:@"."] || [tokenText containsString:@"!"] || 
            [tokenText containsString:@"?"]) {
            
            // For continuations, be less eager to stop
            NSInteger minTokensBeforeStop = isContinuation ? 30 : 20;
            
            if (step > minTokensBeforeStop) {
                // Check if this looks like a complete thought
                NSString *lastFewTokens = [response substringFromIndex:MAX(0, response.length - 50)];
                
                // Don't stop if we're in the middle of code or structured content
                if (![lastFewTokens containsString:@"let "] && 
                    ![lastFewTokens containsString:@"type "] &&
                    ![lastFewTokens containsString:@"match "] &&
                    ![lastFewTokens containsString:@"def "] &&
                    ![lastFewTokens containsString:@"```"]) {
                    printf("\n💭 Natural stopping point reached\n");
                    break;
                }
            }
        }
        
        // Safety check for very long conversations
        if (g_conversationTokens.count > 2000) {
            printf("\n⚠️  Conversation getting very long, consider 'reset'\n");
        }
    }
    
    printf("\n");
    return [response copy];
}

// Main interactive loop
void runSlidingWindowChat() {
    printf("\n🎮 Sliding Window LLaMA Chat\n");
    printf("💡 Type messages for unlimited context conversations\n");
    printf("🪟 Context automatically managed with sliding window\n");
    printf("📝 Commands: 'quit', 'reset', 'status', 'help'\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    char buffer[1024];
    
    while (true) {
        printf("💭 You: ");
        fflush(stdout);
        
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
            printf("\n👋 Goodbye!\n");
            break;
        }
        
        NSString *userInput = [NSString stringWithUTF8String:buffer];
        userInput = [userInput stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        
        if (userInput.length == 0) {
            continue;
        }
        
        // Handle commands
        if ([userInput isEqualToString:@"quit"] || [userInput isEqualToString:@"exit"]) {
            printf("👋 Goodbye!\n");
            break;
        }
        
        if ([userInput isEqualToString:@"reset"]) {
            [g_conversationTokens removeAllObjects];
            printf("🔄 Conversation history reset\n\n");
            continue;
        }
        
        if ([userInput isEqualToString:@"status"]) {
            printf("📊 Conversation status:\n");
            printf("   Total tokens: %lu\n", (unsigned long)g_conversationTokens.count);
            printf("   Sequence limit: %ld\n", g_seqLength);
            printf("   Sliding window: %s\n\n", 
                   g_conversationTokens.count > g_seqLength ? "ACTIVE" : "inactive");
            continue;
        }
        
        if ([userInput isEqualToString:@"help"]) {
            printf("📖 Sliding Window Chat Commands:\n");
            printf("   quit/exit - Exit the chat\n");
            printf("   reset     - Clear conversation history\n");
            printf("   status    - Show conversation statistics\n");
            printf("   help      - Show this help\n");
            printf("   🪟 Unlimited context with automatic sliding window!\n\n");
            continue;
        }
        
        // Generate response
        NSDate *start = [NSDate date];
        NSString *response = generateResponse(userInput);
        double elapsed = [[NSDate date] timeIntervalSinceDate:start];
        
        printf("⏱️  Generated in %.1f seconds (%lu total tokens)\n\n", 
               elapsed, (unsigned long)g_conversationTokens.count);
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: ./sliding_window_llama <model.mlpackage>\n");
            printf("Example: ./sliding_window_llama llama-2-7b-chat.mlpackage\n");
            return 1;
        }
        
        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        
        if (!initializeModel(modelPath)) {
            printf("❌ Failed to initialize model\n");
            return 1;
        }
        
        runSlidingWindowChat();
    }
    
    return 0;
}
