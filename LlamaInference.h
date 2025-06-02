//
//  LlamaInference.h
//  CoreML Llama 2 Demonstrator
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface LlamaInference : NSObject

@property (nonatomic, strong, readonly) MLModel *model;
@property (nonatomic, assign, readonly) BOOL isModelLoaded;

- (instancetype)initWithModelPath:(NSString *)modelPath;
- (void)loadModelWithCompletion:(void (^)(BOOL success, NSError * _Nullable error))completion;
- (void)generateTextWithPrompt:(NSString *)prompt 
                    completion:(void (^)(NSString * _Nullable result, NSError * _Nullable error))completion;
- (void)generateTextWithPrompt:(NSString *)prompt 
                    maxTokens:(NSInteger)maxTokens
                   completion:(void (^)(NSString * _Nullable result, NSError * _Nullable error))completion;

@end

NS_ASSUME_NONNULL_END
