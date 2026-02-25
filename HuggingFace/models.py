from pydantic import BaseModel, Field
from typing import Optional, List

class TextInput(BaseModel):
    text: str = Field(...,min_length=1, examples=["I like this Product !"], description="The input text to be processed.")
    
class BatchTextInput(BaseModel):
    texts: List[str] = Field(...,min_length=1,examples = ["its terrible", "I like this product", "The best product ever"] ,description="the batch of input texts")
    
class sentimentResponse(BaseModel):
    text: str = Field(..., description="The input text that was processed.")
    label: str = Field(..., description="The predicted sentiment label for the input text.") 
    score: float = Field(..., description="The confidence score of the predicted sentiment label.")
    
class BatchSentimentResponse(BaseModel):
    results: List[sentimentResponse] = Field(..., description="A list of sentiment analysis results for each input text in the batch.")
    count : int = Field(..., description="The total number of input texts processed in the batch.")    
       