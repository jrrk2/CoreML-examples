//
//  simple_tokenizer.mm
//  Basic tokenizer for Llama 2 using vocabulary lookup
//

#import <Foundation/Foundation.h>

// Load vocabulary mapping
NSDictionary* loadVocabMapping(NSString *tokenizerPath) {
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:tokenizerPath];
    NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    
    if (!json || error) {
        printf("❌ Failed to load tokenizer: %s\n", error.localizedDescription.UTF8String);
        return nil;
    }
    
    return json[@"model"][@"vocab"];
}

// Simple greedy tokenization - not perfect but works for basic cases
NSArray* tokenizeText(NSString *text, NSDictionary *vocab) {
    NSMutableArray *tokens = [NSMutableArray array];
    
    // Add BOS token
    [tokens addObject:@(1)]; // <s>
    
    // Convert spaces to SentencePiece format
    NSString *spText = [@"▁" stringByAppendingString:
                       [text stringByReplacingOccurrencesOfString:@" " withString:@"▁"]];
    
    printf("🔤 SentencePiece format: %s\n", spText.UTF8String);
    
    // Greedy longest-match tokenization
    NSInteger pos = 0;
    while (pos < spText.length) {
        NSString *bestMatch = nil;
        NSNumber *bestTokenId = nil;
        NSInteger bestLength = 0;
        
        // Try all possible substrings starting at pos, longest first
        for (NSInteger len = MIN(20, spText.length - pos); len >= 1; len--) {
            NSString *substr = [spText substringWithRange:NSMakeRange(pos, len)];
            NSNumber *tokenId = vocab[substr];
            
            if (tokenId && len > bestLength) {
                bestMatch = substr;
                bestTokenId = tokenId;
                bestLength = len;
            }
        }
        
        if (bestMatch) {
            [tokens addObject:bestTokenId];
            printf("📍 '%s' -> %d\n", bestMatch.UTF8String, [bestTokenId intValue]);
            pos += bestLength;
        } else {
            // Fallback: use <unk> and skip character
            [tokens addObject:@(0)]; // <unk>
            printf("❓ Unknown char -> <unk>\n");
            pos += 1;
        }
    }
    
    return [tokens copy];
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: ./simple_tokenizer <tokenizer.json>\n");
            printf("Then enter text to tokenize (Ctrl+D to exit)\n");
            return 1;
        }
        
        NSString *tokenizerPath = [NSString stringWithUTF8String:argv[1]];
        
        printf("🔧 Loading tokenizer from: %s\n", tokenizerPath.UTF8String);
        NSDictionary *vocab = loadVocabMapping(tokenizerPath);
        
        if (!vocab) {
            return 1;
        }
        
        printf("✅ Loaded %lu vocabulary entries\n", (unsigned long)vocab.count);
        printf("\n💬 Enter text to tokenize (press Enter, then Ctrl+D to process):\n");
        
        // Read from stdin
        NSFileHandle *stdin = [NSFileHandle fileHandleWithStandardInput];
        NSData *inputData = [stdin readDataToEndOfFile];
        NSString *inputText = [[NSString alloc] initWithData:inputData encoding:NSUTF8StringEncoding];
        
        if (!inputText || inputText.length == 0) {
            printf("No input received\n");
            return 1;
        }
        
        // Trim whitespace
        inputText = [inputText stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        
        printf("\n🎯 Tokenizing: \"%s\"\n", inputText.UTF8String);
        printf("═══════════════════════════════════════\n");
        
        NSArray *tokens = tokenizeText(inputText, vocab);
        
        printf("═══════════════════════════════════════\n");
        printf("✅ Result: %lu tokens\n", (unsigned long)tokens.count);
        printf("📊 Token IDs: [");
        for (NSInteger i = 0; i < tokens.count; i++) {
            printf("%d", [tokens[i] intValue]);
            if (i < tokens.count - 1) printf(", ");
        }
        printf("]\n");
        
        // For Llama 2 chat format
        printf("\n🗣️  For Llama 2 chat, use: [INST] %s [/INST]\n", inputText.UTF8String);
        
        NSString *chatFormat = [NSString stringWithFormat:@"[INST] %@ [/INST]", inputText];
        NSArray *chatTokens = tokenizeText(chatFormat, vocab);
        
        printf("🗣️  Chat tokens (%lu total): [", (unsigned long)chatTokens.count);
        for (NSInteger i = 0; i < chatTokens.count; i++) {
            printf("%d", [chatTokens[i] intValue]);
            if (i < chatTokens.count - 1) printf(", ");
        }
        printf("]\n");
    }
    
    return 0;
}
