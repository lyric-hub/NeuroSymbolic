import json
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser

_TYPE_DESCRIPTION = (
    "'Vehicle' for any motorised road user (car, truck, motorcycle, bus); "
    "'Pedestrian' for people or cyclists on foot; "
    "'Infrastructure' for static road features (traffic_light, stop_sign, intersection, lane, road)"
)

# 1. Define the strict schema using Pydantic
class SPOTriple(BaseModel):
    subject: str = Field(description="The acting entity, strictly using the tracked ID (e.g., 'Vehicle 4')")
    subject_type: Literal["Vehicle", "Pedestrian", "Infrastructure"] = Field(
        description=f"Entity type of the subject: {_TYPE_DESCRIPTION}"
    )
    predicate: str = Field(description="The action or spatial relationship (e.g., 'tailgating', 'collided_with', 'turning_left')")
    object: str = Field(description="The receiving entity or environment feature (e.g., 'Vehicle 9', 'intersection', 'pedestrian')")
    object_type: Literal["Vehicle", "Pedestrian", "Infrastructure"] = Field(
        description=f"Entity type of the object: {_TYPE_DESCRIPTION}"
    )
    timestamp: float = Field(description="The exact time of the event in seconds")

class SceneGraphOutput(BaseModel):
    triples: List[SPOTriple] = Field(description="A list of all interactions in the scene")

class EntityExtractor:
    """
    Acts as a validation and extraction node in the Sequential Agent Pipeline.
    Takes raw VLM text and forces it into a strict graph-ready schema.
    """
    def __init__(self, model_name: str = "qwen2.5:72b"):
        # Initialize a fast, local LLM via Ollama for parsing
        self.llm = Ollama(model=model_name, temperature=0.0)
        self.parser = JsonOutputParser(pydantic_object=SceneGraphOutput)
        
        # Set up the prompt template that forces the JSON structure
        self.prompt = PromptTemplate(
            template="""
            You are a strict data extraction parser. 
            Extract the traffic interactions from the following Vision Language Model description.
            Convert them into Subject-Predicate-Object triples.
            
            VLM Description: {vlm_text}
            Timestamp of event: {timestamp}
            
            {format_instructions}
            """,
            input_variables=["vlm_text", "timestamp"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        
        # Create the LangChain processing chain
        self.chain = self.prompt | self.llm | self.parser

    def extract_triples(self, raw_vlm_text: str, current_time: float) -> List[dict]:
        """
        Executes the extraction and returns guaranteed, validated dictionaries 
        ready for the Kùzu/Neo4j Graph Database.
        """
        try:
            print(f"Extracting structured entities for timestamp {current_time}...")
            # Run the extraction chain
            result = self.chain.invoke({
                "vlm_text": raw_vlm_text,
                "timestamp": current_time
            })
            
            # Return the validated list of triples
            return result.get("triples", [])
            
        except Exception as e:
            print(f"Failed to parse entities: {e}")
            # Fallback to prevent pipeline crash
            return []

# Example Usage:
if __name__ == "__main__":
    extractor = EntityExtractor()
    mock_vlm_output = "I see Vehicle 4 following very closely behind Vehicle 9 in the heavy rain."
    
    structured_data = extractor.extract_triples(mock_vlm_output, current_time=12.5)
    print(json.dumps(structured_data, indent=2))