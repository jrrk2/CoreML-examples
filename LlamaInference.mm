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
        
        // Configure model for optimal performance
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll; // Use all available compute units
        config.allowLowPrecisionAccumulationOnGPU = YES;
        
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL 
                                           configuration:config 
                                                   error:&error];
        
        dispatch_async(dispatch_get_main_queue(), ^{
            if (model && !error) {
                self.model = model;
                self.isModelLoaded = YES;
                NSLog(@"✅ Model loaded successfully");
                NSLog(@"Model description: %@", model.modelDescription);
                completion(YES, nil);
            } else {
                NSLog(@"❌ Failed to load model: %@", error.localizedDescription);
                completion(NO, error);
            }
        });
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
        
        // Perform prediction
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        id<MLFeatureProvider> output = [self.model predictionFromFeatures:input 
                                                                   options:options 
                                                                     error:&error];
        
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
    // This is where you'll need to adapt based on your model's actual input format
    // Check the model's input description to see what it expects
    
    MLModelDescription *modelDesc = self.model.modelDescription;
    NSLog(@"Input features: %@", modelDesc.inputDescriptionsByName);
    
    // Option 1: If model expects text input directly
    if ([modelDesc.inputDescriptionsByName objectForKey:@"text"]) {
        MLFeatureValue *textFeature = [MLFeatureValue featureValueWithString:text];
        return [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"text": textFeature} 
                                                                 error:error];
    }
    
    // Option 2: If model expects input_ids (tokenized)
    if ([modelDesc.inputDescriptionsByName objectForKey:@"input_ids"]) {
        MLMultiArray *tokenIds = [self tokenizeText:text error:error];
        if (!tokenIds) return nil;
        
        MLFeatureValue *tokenFeature = [MLFeatureValue featureValueWithMultiArray:tokenIds];
        return [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"input_ids": tokenFeature} 
                                                                 error:error];
    }
    
    // Option 3: Check for other input names
    NSString *firstInputName = [modelDesc.inputDescriptionsByName.allKeys firstObject];
    if (firstInputName) {
        // Try string input first
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
        tokenArray[[NSNumber numberWithInteger:i]] = @(tokenId);
    }
    
    return tokenArray;
}

- (NSString *)extractTextFromOutput:(id<MLFeatureProvider>)output error:(NSError **)error {
    // Extract output based on model's output format
    NSDictionary *outputFeatures = [output featureNames];
    NSLog(@"Output features: %@", outputFeatures);
    
    // Common output names for text generation models
    NSArray *possibleOutputNames = @[@"output", @"logits", @"generated_text", @"text_output"];
    
    for (NSString *outputName in possibleOutputNames) {
        MLFeatureValue *feature = [output featureValueForName:outputName];
        if (feature) {
            switch (feature.type) {
                case MLFeatureTypeString:
                    return feature.stringValue;
                    
                case MLFeatureTypeMultiArray: {
                    // Convert logits/token IDs back to text
                    return [self decodeTokens:feature.multiArrayValue];
                }
                    
                default:
                    break;
            }
        }
    }
    
    // If no recognized output, try the first available output
    NSString *firstOutputName = [[output featureNames] firstObject];
    if (firstOutputName) {
        MLFeatureValue *feature = [output featureValueForName:firstOutputName];
        if (feature.type == MLFeatureTypeString) {
            return feature.stringValue;
        }
    }
    
    if (error) {
        *error = [NSError errorWithDomain:@"LlamaInferenceError" 
                                     code:1003 
                                 userInfo:@{NSLocalizedDescriptionKey: @"Unable to extract text from output"}];
    }
    return nil;
}

- (NSString *)decodeTokens:(MLMultiArray *)tokenArray {
    // Simple token decoding - replace with proper detokenization
    NSMutableString *result = [[NSMutableString alloc] init];
    
    NSInteger count = tokenArray.count;
    for (NSInteger i = 0; i < count && i < 100; i++) { // Limit to first 100 tokens
        NSNumber *tokenId = tokenArray[[NSNumber numberWithInteger:i]];
        // This is a placeholder - you'd want proper token-to-text conversion
        [result appendFormat:@"token_%@ ", tokenId];
    }
    
    return [result stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
}

@end
