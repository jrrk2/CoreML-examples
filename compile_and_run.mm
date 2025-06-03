//
//  fresh_compile_and_run.mm
//  Clean CoreML Llama pipeline with proper tokenization
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

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            printf("Usage: ./fresh_compile_and_run <model.mlpackage> <prompt>\n");
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
        NSArray *tokens = @[@(1),      // <s>
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
                           @(29962)];  // ]
        
        printf("🔤 Using %lu tokens\n", (unsigned long)tokens.count);
        
        // Get model input requirements
        NSDictionary *inputDesc = model.modelDescription.inputDescriptionsByName;
        MLFeatureDescription *inputIdsDesc = inputDesc[@"input_ids"];
        NSInteger seqLength = 64; // Default
        
        if (inputIdsDesc.multiArrayConstraint.shape.count >= 2) {
            seqLength = [inputIdsDesc.multiArrayConstraint.shape[1] integerValue];
        }
        printf("📐 Sequence length: %ld\n", seqLength);
        
        // Create input tensors
        NSError *inputError = nil;
        MLMultiArray *inputIds = [[MLMultiArray alloc] initWithShape:@[@(1), @(seqLength)] 
                                                            dataType:MLMultiArrayDataTypeInt32 
                                                               error:&inputError];
        
        MLMultiArray *attentionMask = [[MLMultiArray alloc] initWithShape:@[@(1), @(seqLength)] 
                                                                 dataType:MLMultiArrayDataTypeInt32 
                                                                    error:&inputError];
        
        // Fill tensors
        for (NSInteger i = 0; i < seqLength; i++) {
            NSInteger tokenId = 1; // Default padding with <s>
            NSInteger maskValue = 0; // Default: don't attend to padding
            
            if (i < tokens.count) {
                tokenId = [tokens[i] integerValue];
                maskValue = 1; // Attend to real tokens
            }
            
            inputIds[@[@(0), @(i)]] = @(tokenId);
            attentionMask[@[@(0), @(i)]] = @(maskValue);
        }
        
        // Create feature provider
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
               (unsigned long)tokens.count, seqLength - tokens.count);
        
        // Run inference
        printf("\n🧠 Running inference...\n");
        NSDate *start = [NSDate date];
        
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        id<MLFeatureProvider> result = [model predictionFromFeatures:input 
                                                              options:options 
                                                                error:&loadError];
        
        double inferTime = [[NSDate date] timeIntervalSinceDate:start];
        
        if (!result) {
            printf("❌ Inference failed: %s\n", loadError.localizedDescription.UTF8String);
            return 1;
        }
        
        printf("✅ Inference completed in %.1f seconds\n", inferTime);
        
        // Process output and generate 10 tokens
        printf("\n📊 Generating 10 tokens...\n");
        MLFeatureValue *logits = [result featureValueForName:@"logits"];
        
        if (logits && logits.type == MLFeatureTypeMultiArray) {
            MLMultiArray *logitsArray = logits.multiArrayValue;
            printf("📐 Logits shape: %s\n", [logitsArray.shape.description UTF8String]);
            
            NSInteger vocabSize = [logitsArray.shape[2] integerValue];
            NSMutableArray *generatedTokens = [NSMutableArray array];
            NSMutableString *generatedText = [NSMutableString string];
            
            // Generate 10 tokens
            for (NSInteger tokenStep = 0; tokenStep < 10; tokenStep++) {
                // Find the position to predict from (last real token + generated tokens)
                NSInteger predictionPos = tokens.count - 1 + tokenStep;
                
                if (predictionPos >= seqLength) {
                    printf("⚠️  Reached sequence limit at token %ld\n", tokenStep);
                    break;
                }
                
                // Find best token at this position
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
                
                [generatedTokens addObject:@(bestToken)];
                
                if (vocabulary) {
                    NSString *tokenText = detokenizeWithVocab(bestToken, vocabulary);
                    [generatedText appendString:tokenText];
                    printf("🎯 Token %ld: %ld -> \"%s\" (%.3f)\n", 
                           tokenStep + 1, bestToken, tokenText.UTF8String, bestLogit);
                } else {
                    printf("🎯 Token %ld: %ld (%.3f)\n", tokenStep + 1, bestToken, bestLogit);
                }
            }
            
            printf("\n📝 Complete generated text: \"%s\"\n", generatedText.UTF8String);
            
            // Show the top 3 alternatives for the first token
            NSInteger firstPredPos = tokens.count - 1;
            NSMutableArray *topTokens = [NSMutableArray array];
            
            for (NSInteger i = 0; i < vocabSize; i++) {
                NSNumber *logitVal = logitsArray[@[@(0), @(firstPredPos), @(i)]];
                [topTokens addObject:@{@"token": @(i), @"logit": logitVal}];
            }
            
            [topTokens sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
                return [b[@"logit"] compare:a[@"logit"]];
            }];
            
            printf("\n🌊 Wave function collapsed! But top 3 alternatives for first token were:\n");
            for (NSInteger i = 0; i < MIN(3, topTokens.count); i++) {
                NSDictionary *info = topTokens[i];
                NSInteger tokenId = [info[@"token"] integerValue];
                float logit = [info[@"logit"] floatValue];
                if (vocabulary) {
                    NSString *text = detokenizeWithVocab(tokenId, vocabulary);
                    printf("  %ld. \"%s\" (%.3f)\n", i+1, text.UTF8String, logit);
                }
            }
        }
        
        printf("\n🎉 Complete!\n");
    }
    return 0;
}
