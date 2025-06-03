//
//  method_inspector.mm
//  CoreML API Inspector
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>

void printMethods(Class cls, NSString *className) {
    printf("\n=== %s Methods ===\n", className.UTF8String);
    
    unsigned int methodCount;
    Method *methods = class_copyMethodList(cls, &methodCount);
    
    for (unsigned int i = 0; i < methodCount; i++) {
        Method method = methods[i];
        SEL selector = method_getName(method);
        NSString *methodName = NSStringFromSelector(selector);
        
        // Filter for relevant methods
        if ([methodName containsString:@"State"] || 
            [methodName containsString:@"feature"] ||
            [methodName containsString:@"Value"]) {
            printf("  %s\n", methodName.UTF8String);
        }
    }
    
    free(methods);
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        printf("🔍 CoreML API Inspector\n");
        printf("Checking available methods for stateful models...\n");
        
        // Inspect MLFeatureValue class methods
        printMethods([MLFeatureValue class], @"MLFeatureValue");
        
        // Inspect MLModel class methods  
        printMethods([MLModel class], @"MLModel");
        
        // Check for state-related constants or enums
        printf("\n=== MLFeatureType Values ===\n");
        printf("MLFeatureTypeInvalid = %ld\n", (long)MLFeatureTypeInvalid);
        printf("MLFeatureTypeInt64 = %ld\n", (long)MLFeatureTypeInt64);
        printf("MLFeatureTypeDouble = %ld\n", (long)MLFeatureTypeDouble);
        printf("MLFeatureTypeString = %ld\n", (long)MLFeatureTypeString);
        printf("MLFeatureTypeImage = %ld\n", (long)MLFeatureTypeImage);
        printf("MLFeatureTypeMultiArray = %ld\n", (long)MLFeatureTypeMultiArray);
        printf("MLFeatureTypeDictionary = %ld\n", (long)MLFeatureTypeDictionary);
        printf("MLFeatureTypeSequence = %ld\n", (long)MLFeatureTypeSequence);
        
        // Check if there are state-related feature types
        @try {
            // Try to access potential state-related constants
            printf("Checking for state-related feature types...\n");
            
            // Test if MLFeatureValue has state-related methods
            NSArray *testMethods = @[@"featureValueWithState:",
                                   @"featureValueWithStatefulModel:",
                                   @"stateValue",
                                   @"multiArrayValue"];
            
            for (NSString *methodName in testMethods) {
                SEL selector = NSSelectorFromString(methodName);
                BOOL responds = [MLFeatureValue respondsToSelector:selector];
                printf("  %s: %s\n", methodName.UTF8String, responds ? "YES" : "NO");
            }
            
        } @catch (NSException *exception) {
            printf("Exception during state method check: %s\n", exception.reason.UTF8String);
        }
        
        printf("\n🏁 Inspection complete\n");
    }
    return 0;
}
