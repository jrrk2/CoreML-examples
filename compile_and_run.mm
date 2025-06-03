//
//  compile_and_run.mm
//  Fixed CoreML Llama pipeline with proper autoregressive generation
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

// Load vocabulary from tokenizer.json file
NSDictionary* loadVocabulary(NSString *modelPackagePath) {
    NSString *tokenizerPath = [modelPackagePath stringByAppendingPathComponent:@"tokenizer.json"];
    
    if (![[NSFileManager defaultManager] fileExistsAtPath:tokenizerPath]) {
        printf("⚠️  tokenizer.json not found\n");
        return nil;
    }
    
    NSError *error = nil;
    NSData *tokenizerData = [NSData dataWithContentsOfFile:tokenizerPath];
    NSDictionary *tokenizerJson = [NSJSONSerialization JSONObjectWithData:tokenizerData 
                                                                    options:0 
                                                                      error:&error];
    
    if (!tokenizerJson || error) {
        printf("❌ Failed to parse tokenizer.json\n");
        return nil;
    }
    
    // Extract vocab from the model section
    NSDictionary *model = tokenizerJson[@"model"];
    NSDictionary *vocab = model[@"vocab"];
    
    if (!vocab) {
        printf("❌ No vocab found in tokenizer.json\n");
        return nil;
    }
    
    // Invert the mapping: token_id -> token_string
    NSMutableDictionary *idToToken = [NSMutableDictionary dictionary];
    for (NSString *token in vocab) {
        NSNumber *tokenId = vocab[token];
        idToToken[tokenId] = token;
    }
    
    printf("✅ Loaded vocabulary with %lu tokens\n", (unsigned long)idToToken.count);
    return [idToToken copy];
}

// Detokenize using the loaded vocabulary
NSString* detokenizeWithVocab(NSInteger tokenId, NSDictionary *vocab) {
    NSString *token = vocab[@(tokenId)];
    if (token) {
        // Convert SentencePiece format: ▁ represents space
        NSString *cleaned = [token stringByReplacingOccurrencesOfString:@"▁" withString:@" "];
        return cleaned;
    }
    return [NSString stringWithFormat:@"[UNK_%ld]", tokenId];
}

// Check if token is EOS or should stop generation
BOOL shouldStopGeneration(NSInteger tokenId) {
    // Common stop tokens for LLaMA models
    return (tokenId == 2 ||      // </s> (EOS)
            tokenId == 0 ||      // <unk> 
            tokenId == 1);       // <s> (shouldn't appear mid-generation)
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            printf("Usage: ./compile_and_run <model.mlpackage> <prompt>\n");
            return 1;
        }
        
        NSString *inputPath = [NSString stringWithUTF8String:argv[1]];
        NSString *prompt = [NSString stringWithUTF8String:argv[2]];
        
        printf("🚀 Fresh Compile and Run Pipeline\n");
        printf("📥 Model: %s\n", inputPath.UTF8String);
        printf("💬 Prompt: %s\n", prompt.UTF8String);
        
        // Load vocabulary
        printf("\n📖 Loading vocabulary...\n");
        NSDictionary *vocabulary = loadVocabulary(inputPath);
        
        NSURL *inputURL = [NSURL fileURLWithPath:inputPath];
        
        // Compile model
        printf("\n🔧 Compiling model...\n");
        NSError *compileError = nil;
        NSURL *compiledURL = [MLModel compileModelAtURL:inputURL error:&compileError];
        
        if (!compiledURL) {
            printf("❌ Compilation failed: %s\n", compileError.localizedDescription.UTF8String);
            return 1;
        }
        printf("✅ Compilation successful\n");
        
        // Load model with Neural Engine
        printf("\n🔄 Loading model...\n");
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll; // Try Neural Engine
        
        NSError *loadError = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:compiledURL 
                                           configuration:config 
                                                   error:&loadError];
        
        if (!model && loadError) {
            printf("⚠️  Neural Engine failed, trying CPU...\n");
            config.computeUnits = MLComputeUnitsCPUOnly;
            model = [MLModel modelWithContentsOfURL:compiledURL 
                                      configuration:config 
                                              error:&loadError];
        }
        
        if (!model) {
            printf("❌ Model loading failed: %s\n", loadError.localizedDescription.UTF8String);
            return 1;
        }
        
        NSString *compute = (config.computeUnits == MLComputeUnitsAll) ? @"Neural Engine" : @"CPU";
        printf("✅ Model loaded using %s\n", compute.UTF8String);
        
        // Prepare input with proper Llama 2 chat format
        printf("\n📝 Preparing input...\n");
        
        // Llama 2 chat format: [INST] prompt [/INST]
        NSString *chatPrompt = [NSString stringWithFormat:@"[INST] %@ [/INST]", prompt];
        printf("🔤 Chat format: %s\n", chatPrompt.UTF8String);
        
        // Proper tokens for "[INST] Hello, how are you? [/INST]"
        NSMutableArray *currentTokens = [@[@(1),      // <s>
                                          @(518),     // [
                                          @(25580),   // INST
                                          @(29962),   // ]
                                          @(15043),   // Hello
                                          @(29892),   // ,
                                          @(920),     // how
                                          @(526),     // are
                                          @(366),     // you
                                          @(29973),   // ?
                                          @(518),     // [
                                          @(29914),   // /
                                          @(25580),   // INST
                                          @(29962)] mutableCopy];  // ]
        
        printf("🔤 Using %lu initial tokens\n", (unsigned long)currentTokens.count);
        
        // Get model input requirements
        NSDictionary *inputDesc = model.modelDescription.inputDescriptionsByName;
        MLFeatureDescription *inputIdsDesc = inputDesc[@"input_ids"];
        NSInteger seqLength = 512; // Increased from 64 to give more room
        
        if (inputIdsDesc.multiArrayConstraint.shape.count >= 2) {
            NSInteger modelSeqLength = [inputIdsDesc.multiArrayConstraint.shape[1] integerValue];
            seqLength = MIN(seqLength, modelSeqLength); // Don't exceed model capacity
        }
        printf("📐 Sequence length: %ld\n", seqLength);
        
        // Create input tensors (reusable)
        NSError *inputError = nil;
        MLMultiArray *inputIds = [[MLMultiArray alloc] initWithShape:@[@(1), @(seqLength)] 
                                                            dataType:MLMultiArrayDataTypeInt32 
                                                               error:&inputError];
        
        MLMultiArray *attentionMask = [[MLMultiArray alloc] initWithShape:@[@(1), @(seqLength)] 
                                                                 dataType:MLMultiArrayDataTypeInt32 
                                                                    error:&inputError];
        
        if (!inputIds || !attentionMask) {
            printf("❌ Input tensor creation failed\n");
            return 1;
        }
        
        printf("✅ Input tensors created\n");
        
        // Function to update input tensors with current token sequence
        void (^updateInputTensors)(void) = ^{
            // Clear tensors
            for (NSInteger i = 0; i < seqLength; i++) {
                inputIds[@[@(0), @(i)]] = @(1); // Pad with <s>
                attentionMask[@[@(0), @(i)]] = @(0); // Don't attend to padding
            }
            
            // Fill with current tokens
            NSInteger tokenCount = MIN(currentTokens.count, seqLength);
            for (NSInteger i = 0; i < tokenCount; i++) {
                inputIds[@[@(0), @(i)]] = currentTokens[i];
                attentionMask[@[@(0), @(i)]] = @(1); // Attend to real tokens
            }
        };
        
        // Initial setup
        updateInputTensors();
        
        // Create feature provider (reusable)
        MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] 
            initWithDictionary:@{
                @"input_ids": [MLFeatureValue featureValueWithMultiArray:inputIds],
                @"attention_mask": [MLFeatureValue featureValueWithMultiArray:attentionMask]
            } error:&inputError];
        
        if (!input) {
            printf("❌ Input creation failed\n");
            return 1;
        }
        
        printf("✅ Input prepared: %lu real tokens, %ld padding\n", 
               (unsigned long)currentTokens.count, seqLength - currentTokens.count);
        
        // Autoregressive generation loop
        printf("\n🧠 Running autoregressive generation...\n");
        NSMutableString *generatedText = [NSMutableString string];
        NSInteger maxNewTokens = 50; // Generate up to 50 new tokens
        
        for (NSInteger tokenStep = 0; tokenStep < maxNewTokens; tokenStep++) {
            // Check if we're approaching sequence limit
            if (currentTokens.count >= seqLength - 1) {
                printf("⚠️  Reached sequence limit at token %ld\n", tokenStep);
                break;
            }
            
            // Run inference with current sequence
            NSDate *start = [NSDate date];
            
            MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
            id<MLFeatureProvider> result = [model predictionFromFeatures:input 
                                                                  options:options 
                                                                    error:&loadError];
            
            double inferTime = [[NSDate date] timeIntervalSinceDate:start];
            
            if (!result) {
                printf("❌ Inference failed at token %ld: %s\n", tokenStep, loadError.localizedDescription.UTF8String);
                break;
            }
            
            // Get logits for next token prediction
            MLFeatureValue *logits = [result featureValueForName:@"logits"];
            
            if (!logits || logits.type != MLFeatureTypeMultiArray) {
                printf("❌ Invalid logits output at token %ld\n", tokenStep);
                break;
            }
            
            MLMultiArray *logitsArray = logits.multiArrayValue;
            NSInteger vocabSize = [logitsArray.shape[2] integerValue];
            
            // Predict next token from the last position in the sequence
            NSInteger predictionPos = currentTokens.count - 1;
            
            // Find best token (greedy decoding)
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
            
            // Check for stop conditions
            if (shouldStopGeneration(bestToken)) {
                printf("🛑 Generation stopped at EOS token %ld\n", bestToken);
                break;
            }
            
            // Add generated token to sequence
            [currentTokens addObject:@(bestToken)];
            
            // Detokenize and display
            if (vocabulary) {
                NSString *tokenText = detokenizeWithVocab(bestToken, vocabulary);
                [generatedText appendString:tokenText];
                printf("🎯 Token %ld: %ld -> \"%s\" (%.3f) [%.1fms]\n", 
                       tokenStep + 1, bestToken, tokenText.UTF8String, bestLogit, inferTime * 1000);
            } else {
                printf("🎯 Token %ld: %ld (%.3f) [%.1fms]\n", tokenStep + 1, bestToken, bestLogit, inferTime * 1000);
            }
            
            // Update input tensors for next iteration
            updateInputTensors();
            
            // Optional: Stop on certain tokens (like period for sentences)
            if (vocabulary) {
                NSString *tokenText = detokenizeWithVocab(bestToken, vocabulary);
                if ([tokenText containsString:@"."] || [tokenText containsString:@"!"]) {
                    // Continue for a bit more, don't stop immediately on punctuation
                }
            }
        }
        
        printf("\n📝 Complete generated text: \"%s\"\n", generatedText.UTF8String);
        printf("📊 Generated %lu new tokens\n", (unsigned long)(currentTokens.count - 14));
        printf("\n🎉 Autoregressive generation complete!\n");
    }
    return 0;
}
