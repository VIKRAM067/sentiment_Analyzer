from transformers import pipeline

# Module-level variable — loaded once, reused forever
_pipeline = None # the underscore indicates "private" — not to be used outside this module

def get_pipeline():
    # global indicates that we want to modify the module-level variable which is above, not create a new local variable
    global _pipeline 
    if _pipeline is None:
        print("Loading model... (first request only)")
        
        # loading the model , loaded only once, and then reused for all subsequent requests
        _pipeline = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _pipeline