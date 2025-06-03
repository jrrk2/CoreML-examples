//
//  interactive_llama_engine.mm
//  Line-by-line inference engine with proper SentencePiece tokenization
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <iostream>
#include <string>

// Global state
static NSDictionary *g_vocabulary = nil;
static NSDictionary *g_invVocabulary = nil; // token_id -> token_string
static MLModel *g_model = nil;
static MLMultiArray *g_inputIds = nil;
static MLMultiArray *g_attentionMask = nil;
static MLDictionaryFeatureProvider *g_input = nil;
static NSInteger g_seqLength = 64;

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
    
    // Create inverse mapping: token_id -> token_string
    NSMutableDictionary *idToToken = [NSMutableDictionary dictionary];
    for (NSString *token in g_vocabulary) {
        NSNumber *tokenId = g_vocabulary[token];
        idToToken[tokenId] = token;
    }
    g_invVocabulary = [idToToken copy];
    
    printf("✅ Loaded vocabulary with %lu tokens\n", (unsigned long)g_vocabulary.count);
    return YES;
}

// SentencePiece tokenization using your greedy longest-match algorithm
NSArray* tokenizeText(NSString *text) {
    NSMutableArray *tokens = [NSMutableArray array];
    
    // Add BOS token for start of sequence
    [tokens addObject:@(1)]; // <s>
    
    // Convert to SentencePiece format (spaces become ▁)
    NSString *spText = [@"▁" stringByAppendingString:
                       [text stringByReplacingOccurrencesOfString:@" " withString:@"▁"]];
    
    // Greedy longest-match tokenization
    NSInteger pos = 0;
    while (pos < spText.length) {
        NSString *bestMatch = nil;
        NSNumber *bestTokenId = nil;
        NSInteger bestLength = 0;
        
        // Try all possible substrings starting at pos, longest first
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
            // Fallback: use <unk> and skip character
            [tokens addObject:@(0)]; // <unk>
            pos += 1;
        }
    }
    
    return [tokens copy];
}

// Wrap text in LLaMA chat format and tokenize
NSArray* tokenizeChatInput(NSString *userInput) {
    NSString *chatFormat = [NSString stringWithFormat:@"[INST] %@ [/INST]", userInput];
    return tokenizeText(chatFormat);
}

// Detokenize token ID back to text
NSString* detokenize(NSInteger tokenId) {
    NSString *token = g_invVocabulary[@(tokenId)];
    if (token) {
        // Convert SentencePiece format: ▁ represents space
        return [token stringByReplacingOccurrencesOfString:@"▁" withString:@" "];
    }
    return [NSString stringWithFormat:@"[UNK_%ld]", tokenId];
}

// Check if generation should stop
BOOL shouldStopGeneration(NSInteger tokenId) {
    return (tokenId == 2 ||      // </s> (EOS)
            tokenId == 0 ||      // <unk>
            tokenId == 13);      // \n (newline) - stop on line breaks for better interaction
}

// Initialize model once at startup
BOOL initializeModel(NSString *modelPath) {
    printf("🔧 Initializing LLaMA inference engine...\n");
    printf("📥 Model: %s\n", modelPath.UTF8String);
    
    // Load vocabulary
    if (!loadVocabulary(modelPath)) {
        return NO;
    }
    
    NSURL *inputURL = [NSURL fileURLWithPath:modelPath];
    
    // Compile model
    printf("⚙️  Compiling model...\n");
    NSError *compileError = nil;
    NSURL *compiledURL = [MLModel compileModelAtURL:inputURL error:&compileError];
    
    if (!compiledURL) {
        printf("❌ Compilation failed: %s\n", compileError.localizedDescription.UTF8String);
        return NO;
    }
    
    // Load model
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
    
    // Auto-detect sequence length from model
    NSDictionary *inputDesc = g_model.modelDescription.inputDescriptionsByName;
    MLFeatureDescription *inputIdsDesc = inputDesc[@"input_ids"];
    if (inputIdsDesc.multiArrayConstraint.shape.count >= 2) {
        g_seqLength = [inputIdsDesc.multiArrayConstraint.shape[1] integerValue];
    }
    printf("📐 Detected sequence length: %ld\n", g_seqLength);
    
    // Create reusable input tensors
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
    
    printf("🎉 Initialization complete! Sequence length: %ld\n", g_seqLength);
    return YES;
}

// Update input tensors with current token sequence
void updateInputTensors(NSArray *tokens) {
    // Clear tensors
    for (NSInteger i = 0; i < g_seqLength; i++) {
        g_inputIds[@[@(0), @(i)]] = @(1); // Pad with <s>
        g_attentionMask[@[@(0), @(i)]] = @(0); // Don't attend to padding
    }
    
    // Fill with current tokens
    NSInteger tokenCount = MIN(tokens.count, g_seqLength);
    for (NSInteger i = 0; i < tokenCount; i++) {
        g_inputIds[@[@(0), @(i)]] = tokens[i];
        g_attentionMask[@[@(0), @(i)]] = @(1); // Attend to real tokens
    }
}

// Generate response using autoregressive inference
NSString* generateResponse(NSString *userInput, NSInteger maxTokens) {
    // Tokenize the chat input
    NSArray *inputTokens = tokenizeChatInput(userInput);
    
    printf("🔤 Tokenized '%s' to %lu tokens\n", userInput.UTF8String, (unsigned long)inputTokens.count);
    
    // Check if input fits in sequence (more reasonable limits)
    NSInteger availableSpace = g_seqLength - inputTokens.count;
    if (availableSpace < 10) {
        // Only truncate if we really don't have space
        NSInteger maxInputTokens = g_seqLength - 15; // Leave 15 tokens for response
        printf("⚠️  Input too long (%lu tokens), truncating to %ld...\n", 
               (unsigned long)inputTokens.count, maxInputTokens);
        inputTokens = [inputTokens subarrayWithRange:NSMakeRange(0, maxInputTokens)];
        availableSpace = 15;
    }
    
    // Adjust maxTokens to fit in available space
    maxTokens = MIN(maxTokens, availableSpace - 2);
    printf("🎯 Generating up to %ld tokens\n", maxTokens);
    
    NSMutableArray *currentTokens = [inputTokens mutableCopy];
    NSMutableString *response = [NSMutableString string];
    
    printf("🤖 Assistant: ");
    fflush(stdout);
    
    // Autoregressive generation loop
    for (NSInteger step = 0; step < maxTokens; step++) {
        if (currentTokens.count >= g_seqLength - 1) {
            printf("\n⚠️  Sequence limit reached\n");
            break;
        }
        
        // Update tensors with current sequence
        updateInputTensors(currentTokens);
        
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
        
        // Extract logits and find best next token
        MLFeatureValue *logits = [result featureValueForName:@"logits"];
        if (!logits || logits.type != MLFeatureTypeMultiArray) {
            printf("\n❌ Invalid logits\n");
            break;
        }
        
        MLMultiArray *logitsArray = logits.multiArrayValue;
        NSInteger vocabSize = [logitsArray.shape[2] integerValue];
        NSInteger predictionPos = currentTokens.count - 1;
        
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
        
        // Add token to sequence
        [currentTokens addObject:@(bestToken)];
        
        // Detokenize and stream output
        NSString *tokenText = detokenize(bestToken);
        [response appendString:tokenText];
        printf("%s", tokenText.UTF8String);
        fflush(stdout);
        
        // Stop on natural sentence boundaries for better interaction
        if ([tokenText containsString:@"."] || [tokenText containsString:@"!"] || 
            [tokenText containsString:@"?"] || [tokenText containsString:@"\n"]) {
            // Could add logic to stop after complete sentences
        }
    }
    
    printf("\n");
    return [response copy];
}

// Main interactive loop
void runInteractiveChat() {
    printf("\n🎮 Interactive LLaMA Chat\n");
    printf("💡 Type your messages and press Enter (Ctrl+C to exit)\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    char buffer[1024];
    
    while (true) {
        printf("💭 You: ");
        fflush(stdout);
        
        // Read line from stdin
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
            printf("\n👋 Goodbye!\n");
            break;
        }
        
        // Convert to NSString and trim
        NSString *userInput = [NSString stringWithUTF8String:buffer];
        userInput = [userInput stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        
        if (userInput.length == 0) {
            continue;
        }
        
        // Special commands
        if ([userInput isEqualToString:@"quit"] || [userInput isEqualToString:@"exit"]) {
            printf("👋 Goodbye!\n");
            break;
        }
        
        if ([userInput isEqualToString:@"help"]) {
            printf("📖 Commands:\n");
            printf("  quit/exit - Exit the chat\n");
            printf("  help      - Show this help\n");
            printf("  Otherwise, just type your message!\n\n");
            continue;
        }
        
        // Generate response
        NSDate *start = [NSDate date];
        NSString *response = generateResponse(userInput, 30); // Reduced from 50 to 30
        double elapsed = [[NSDate date] timeIntervalSinceDate:start];
        
        printf("⏱️  Generated in %.1f seconds\n\n", elapsed);
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: ./interactive_llama_engine <model.mlpackage>\n");
            printf("Example: ./interactive_llama_engine llama-2-7b-chat.mlpackage\n");
            return 1;
        }
        
        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        
        // Initialize the model
        if (!initializeModel(modelPath)) {
            printf("❌ Failed to initialize model\n");
            return 1;
        }
        
        // Start interactive chat
        runInteractiveChat();
    }
    
    return 0;
}
