//
//  LlamaInference.mm
//  CoreML Llama 2 Demonstrator
//

#import "LlamaInference.h"
#import <vector>
#import <string>
#import <sstream>

@interface LlamaInference ()
@property (nonatomic, strong) NSString *modelPath;
@property (nonatomic, strong) MLModel *model;
@property (nonatomic, strong) MLState *modelState; // Add state property
@property (nonatomic, assign) BOOL isModelLoaded;
@end

@implementation LlamaInference

- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (self) {
        _modelPath = modelPath;
        _isModelLoaded = NO;
    }
    return self;
}

- (void)loadModelWithCompletion:(void (^)(BOOL success, NSError * _Nullable error))completion {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        NSError *error = nil;
        NSURL *modelURL = [NSURL fileURLWithPath:self.modelPath];
        
        NSLog(@"📁 Model path: %@", modelURL);
        NSLog(@"💾 Loading model weights... this may take 2-5 minutes");
        NSLog(@"⏱️  Loading started at: %@", [NSDate date]);
        
        // Configure model - try CPU only to avoid GPU bytecode issues
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUOnly; // CPU only to avoid GPU bytecode error
        
        NSLog(@"🔧 Configuration: computeUnits=CPU_ONLY (avoiding GPU bytecode issue)");
        NSLog(@"🚀 Calling MLModel modelWithContentsOfURL...");
        
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL 
                                           configuration:config 
                                                   error:&error];
        
        NSLog(@"⏱️  Model loading completed at: %@", [NSDate date]);
        
        // Call completion on the same background thread - don't switch to main queue
        if (model && !error) {
            self.model = model;
            
            // Create state for stateful models
            MLState *state = [model newState];
            if (state) {
                self.modelState = state;
                NSLog(@"✅ Model state created: %@", state);
            } else {
                NSLog(@"⚠️  No state created - might not be a stateful model");
            }
            
            self.isModelLoaded = YES;
            NSLog(@"✅ Model loaded successfully");
            NSLog(@"Model description: %@", model.modelDescription);
            completion(YES, nil);
        } else {
            NSLog(@"❌ Failed to load model: %@", error.localizedDescription);
            completion(NO, error);
        }
    });
}

- (void)generateTextWithPrompt:(NSString *)prompt 
                    completion:(void (^)(NSString * _Nullable result, NSError * _Nullable error))completion {
    [self generateTextWithPrompt:prompt maxTokens:100 completion:completion];
}

- (void)generateTextWithPrompt:(NSString *)prompt 
                    maxTokens:(NSInteger)maxTokens
                   completion:(void (^)(NSString * _Nullable result, NSError * _Nullable error))completion {
    
    if (!self.isModelLoaded) {
        NSError *error = [NSError errorWithDomain:@"LlamaInferenceError" 
                                             code:1001 
                                         userInfo:@{NSLocalizedDescriptionKey: @"Model not loaded"}];
        completion(nil, error);
        return;
    }
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        NSError *error = nil;
        
        // Format prompt for Llama 2 Chat
        NSString *formattedPrompt = [self formatChatPrompt:prompt];
        
        // Prepare input - this will vary based on your specific model's input format
        // Most CoreML Llama models expect either:
        // 1. "input_ids" as MLMultiArray
        // 2. "text" as NSString
        // 3. Custom input format
        
        MLDictionaryFeatureProvider *input = [self prepareInputWithText:formattedPrompt error:&error];
        if (!input) {
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(nil, error);
            });
            return;
        }
        
        // Perform prediction - use stateful method if we have state
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        id<MLFeatureProvider> output;
        
        if (self.modelState) {
            NSLog(@"🔄 Using stateful prediction method");
            output = [self.model predictionFromFeatures:input 
                                             usingState:self.modelState 
                                                  error:&error];
        } else {
            NSLog(@"🔄 Using regular prediction method");
            output = [self.model predictionFromFeatures:input 
                                               options:options 
                                                 error:&error];
        }
        
        if (error) {
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(nil, error);
            });
            return;
        }
        
        // Extract and process output
        NSString *result = [self extractTextFromOutput:output error:&error];
        
        dispatch_async(dispatch_get_main_queue(), ^{
            completion(result, error);
        });
    });
}

#pragma mark - Private Methods

- (NSString *)formatChatPrompt:(NSString *)userMessage {
    // Llama 2 Chat format
    return [NSString stringWithFormat:@"<s>[INST] %@ [/INST]", userMessage];
}

- (MLDictionaryFeatureProvider *)prepareInputWithText:(NSString *)text error:(NSError **)error {
    MLModelDescription *modelDesc = self.model.modelDescription;
    NSLog(@"Input features: %@", modelDesc.inputDescriptionsByName);
    
    // For stateful Mistral model: simple inputs only, state managed separately
    if ([modelDesc.inputDescriptionsByName objectForKey:@"inputIds"] && 
        [modelDesc.inputDescriptionsByName objectForKey:@"causalMask"]) {
        
        // Simple single token approach for demo
        NSInteger firstChar = [text length] > 0 ? [text characterAtIndex:0] : 72; // 'H'
        NSInteger tokenId = (firstChar % 32000) + 1; // Map to vocab range
        
        NSLog(@"🔤 Converting '%@' -> first char '%c' -> token ID %ld", 
              text, (char)firstChar, tokenId);
        
        // Create inputIds [1, 1] - single token
        MLMultiArray *inputIds = [[MLMultiArray alloc] initWithShape:@[@(1), @(1)] 
                                                            dataType:MLMultiArrayDataTypeInt32 
                                                               error:error];
        if (!inputIds) return nil;
        inputIds[@[@(0), @(0)]] = @(tokenId);
        
        // Create causalMask [1, 1, 1, 1] - 4D tensor as expected
        MLMultiArray *causalMask = [[MLMultiArray alloc] initWithShape:@[@(1), @(1), @(1), @(1)] 
                                                              dataType:MLMultiArrayDataTypeFloat16 
                                                                 error:error];
        if (!causalMask) return nil;
        causalMask[@[@(0), @(0), @(0), @(0)]] = @(1.0); // Allow this position
        
        NSLog(@"📊 Created inputIds shape: %@, causalMask shape: %@", 
              inputIds.shape, causalMask.shape);
        
        // Return simple feature provider - NO explicit state inputs needed!
        return [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
            @"inputIds": [MLFeatureValue featureValueWithMultiArray:inputIds],
            @"causalMask": [MLFeatureValue featureValueWithMultiArray:causalMask]
        } error:error];
    }
    
    // Fallback for other model types
    NSString *firstInputName = [modelDesc.inputDescriptionsByName.allKeys firstObject];
    if (firstInputName) {
        MLFeatureValue *feature = [MLFeatureValue featureValueWithString:text];
        return [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{firstInputName: feature} 
                                                                 error:error];
    }
    
    if (error) {
        *error = [NSError errorWithDomain:@"LlamaInferenceError" 
                                     code:1002 
                                 userInfo:@{NSLocalizedDescriptionKey: @"Unable to determine input format"}];
    }
    return nil;
}

- (MLMultiArray *)tokenizeText:(NSString *)text error:(NSError **)error {
    // Simple tokenization - in practice you'd want proper tokenization
    // This is a placeholder that creates dummy token IDs
    NSArray<NSString *> *words = [text componentsSeparatedByString:@" "];
    NSInteger tokenCount = words.count;
    
    NSArray<NSNumber *> *shape = @[@(1), @(tokenCount)]; // [batch_size, sequence_length]
    MLMultiArray *tokenArray = [[MLMultiArray alloc] initWithShape:shape 
                                                          dataType:MLMultiArrayDataTypeInt32 
                                                             error:error];
    if (!tokenArray) return nil;
    
    // Fill with dummy token IDs (1-1000)
    for (NSInteger i = 0; i < tokenCount; i++) {
        NSInteger tokenId = (words[i].hash % 1000) + 1;
        NSArray<NSNumber *> *key = @[@(0), @(i)]; // [batch_index, token_index]
        tokenArray[key] = @(tokenId);
    }
    
    return tokenArray;
}

- (NSString *)extractTextFromOutput:(id<MLFeatureProvider>)output error:(NSError **)error {
    // Extract logits from the Mistral model output
    MLFeatureValue *logitsFeature = [output featureValueForName:@"logits"];
    
    if (logitsFeature && logitsFeature.type == MLFeatureTypeMultiArray) {
        MLMultiArray *logits = logitsFeature.multiArrayValue;
        NSLog(@"📊 Logits shape: %@", logits.shape);
        
        // For demo: just take the last token's logits and find the max probability
        NSArray<NSNumber *> *shape = logits.shape;
        if (shape.count >= 3) { // [batch, sequence, vocab]
            NSInteger batchSize = [shape[0] integerValue];
            NSInteger seqLength = [shape[1] integerValue];
            NSInteger vocabSize = [shape[2] integerValue];
            
            NSLog(@"📐 Batch: %ld, Sequence: %ld, Vocab: %ld", batchSize, seqLength, vocabSize);
            
            // Get the last token's logits (most recent prediction)
            NSInteger lastTokenIndex = seqLength - 1;
            float maxLogit = -INFINITY;
            NSInteger bestTokenId = 0;
            
            // Find token with highest probability (greedy decoding)
            for (NSInteger v = 0; v < MIN(vocabSize, 1000); v++) { // Check first 1000 tokens
                NSNumber *logitValue = logits[@[@(0), @(lastTokenIndex), @(v)]];
                float logit = [logitValue floatValue];
                if (logit > maxLogit) {
                    maxLogit = logit;
                    bestTokenId = v;
                }
            }
            
            NSLog(@"🎯 Best token ID: %ld with logit: %.3f", bestTokenId, maxLogit);
            
            // Convert token ID back to text (placeholder - needs real detokenizer)
            return [NSString stringWithFormat:@"Generated token ID: %ld (logit: %.3f)", bestTokenId, maxLogit];
        }
    }
    
    // Fallback to original logic
    NSSet<NSString *> *outputFeatureNames = [output featureNames];
    NSLog(@"Output features: %@", outputFeatureNames);
    
    if (error) {
        *error = [NSError errorWithDomain:@"LlamaInferenceError" 
                                     code:1003 
                                 userInfo:@{NSLocalizedDescriptionKey: @"Unable to extract meaningful text from logits"}];
    }
    return @"[Logits output - needs proper detokenization]";
}

- (NSString *)decodeTokens:(MLMultiArray *)tokenArray {
    // Simple token decoding - replace with proper detokenization
    NSMutableString *result = [[NSMutableString alloc] init];
    
    NSInteger count = tokenArray.count;
    for (NSInteger i = 0; i < count && i < 100; i++) { // Limit to first 100 tokens
        NSArray<NSNumber *> *key = @[@(0), @(i)]; // [batch_index, token_index]
        NSNumber *tokenId = tokenArray[key];
        // This is a placeholder - you'd want proper token-to-text conversion
        [result appendFormat:@"token_%@ ", tokenId];
    }
    
    return [result stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
}

@end
