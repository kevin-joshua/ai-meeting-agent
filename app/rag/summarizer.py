import os
from groq import Groq
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import re
from dotenv import load_dotenv
load_dotenv()

class MeetingTranscriptionSummarizer:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "llama3-70b-8192",
                 max_tokens: int = 2048,
                 temperature: float = 0.1):
        """
        Initialize the Meeting Transcription Summarizer with Groq
        
        Args:
            api_key: Groq API key (if None, will use GROQ_API_KEY env var)
            model: Groq model to use for summarization
            max_tokens: Maximum tokens for the response
            temperature: Temperature for response generation (lower = more focused)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key must be provided or set in GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    def create_summarization_prompt(self, transcription: str, context: Dict[str, Any] = None) -> str:
        """
        Create a comprehensive prompt for meeting summarization
        
        Args:
            transcription: The full meeting transcription text
            context: Additional context about the meeting
            
        Returns:
            Formatted prompt string
        """
        context = context or {}
        
        prompt = f"""You are an expert meeting summarizer. Analyze the following meeting transcription and provide a comprehensive, structured summary.

TRANSCRIPTION:
{transcription}

Please provide a detailed summary in the following JSON format:

{{
    "meeting_title": "Extract or infer the meeting title/topic from the content",
    "date_time": "Extract date and time if mentioned, otherwise 'Not specified'",
    "participants": [
        {{
            "name": "Participant name (if identifiable)",
            "role": "Their role or title (if mentioned)"
        }}
    ],
    "high_level_summary": "A concise 2-3 sentence overview of the entire meeting",
    "key_discussion_points": [
        "Main topic or agenda item 1",
        "Main topic or agenda item 2",
        "Additional important discussion points..."
    ],
    "decisions_made": [
        "Clear decision 1 with context",
        "Clear decision 2 with context",
        "Additional decisions..."
    ],
    "action_items": [
        {{
            "task": "Description of the action item",
            "assignee": "Person responsible (if mentioned)",
            "due_date": "Due date if specified, otherwise 'Not specified'",
            "priority": "High/Medium/Low (infer from context)"
        }}
    ],
    "open_questions": [
        "Unresolved question or issue 1",
        "Unresolved question or issue 2",
        "Additional open items..."
    ],
    "sentiment_analysis": {{
        "overall_tone": "Professional/Casual/Tense/Collaborative/etc.",
        "engagement_level": "High/Medium/Low",
        "conflict_indicators": "Any signs of disagreement or tension",
        "consensus_level": "High/Medium/Low agreement among participants"
    }},
    "meeting_effectiveness": {{
        "clarity_of_objectives": "High/Medium/Low",
        "decision_making_efficiency": "High/Medium/Low",
        "follow_up_clarity": "High/Medium/Low"
    }},
    "additional_insights": [
        "Any other relevant observations or patterns",
        "Suggestions for improvement",
        "Notable quotes or key statements"
    ]
}}

IMPORTANT INSTRUCTIONS:
1. Extract information directly from the transcription - don't make assumptions
2. If information is not available, use "Not specified" or "Not mentioned"
3. Be precise and factual in your analysis
4. Focus on actionable items and clear outcomes
5. Identify speakers by context clues if names aren't explicitly mentioned
6. Infer sentiment and tone from the language used and interaction patterns
7. Ensure all JSON fields are properly formatted
8. If the transcription seems incomplete or unclear, note this in additional_insights

Provide only the JSON response, no additional text."""

        return prompt
    
    def preprocess_transcription(self, transcription: str) -> str:
        """
        Clean and preprocess the transcription for better LLM processing
        
        Args:
            transcription: Raw transcription text
            
        Returns:
            Cleaned transcription
        """
        if not transcription:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', transcription)
        text = re.sub(r' +', ' ', text)
        
        # Clean up common transcription artifacts
        text = re.sub(r'\[inaudible\]|\[unclear\]|\[crosstalk\]', '[unclear audio]', text, flags=re.IGNORECASE)
        
        # Normalize speaker indicators
        text = re.sub(r'Speaker \d+:', 'Speaker:', text)
        text = re.sub(r'Person \d+:', 'Speaker:', text)
        
        return text.strip()
    
    def chunk_long_transcription(self, transcription: str, max_chunk_size: int = 15000) -> List[str]:
        """
        Split very long transcriptions into manageable chunks
        
        Args:
            transcription: Full transcription text
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of transcription chunks
        """
        if len(transcription) <= max_chunk_size:
            return [transcription]
        
        chunks = []
        words = transcription.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize_transcription(self, 
                              transcription: str, 
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Summarize a meeting transcription using Groq LLM
        
        Args:
            transcription: Full meeting transcription
            context: Additional context about the meeting
            
        Returns:
            Structured summary dictionary
        """
        try:
            # Preprocess the transcription
            cleaned_transcription = self.preprocess_transcription(transcription)
            
            if not cleaned_transcription:
                raise ValueError("Empty transcription after preprocessing")
            
            # Handle very long transcriptions
            chunks = self.chunk_long_transcription(cleaned_transcription)
            
            if len(chunks) == 1:
                # Single chunk processing
                return self._process_single_chunk(chunks[0], context)
            else:
                # Multi-chunk processing
                return self._process_multiple_chunks(chunks, context)
                
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _process_single_chunk(self, transcription: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single transcription chunk"""
        prompt = self.create_summarization_prompt(transcription, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert meeting summarizer. Provide accurate, structured summaries in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            summary_text = response.choices[0].message.content
            summary = json.loads(summary_text)
            
            # Add metadata
            summary["processing_info"] = {
                "status": "success",
                "model_used": self.model,
                "chunks_processed": 1,
                "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
                "processing_time": datetime.now().isoformat()
            }
            
            return summary
            
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error_type": "json_parse_error",
                "error_message": f"Failed to parse LLM response as JSON: {str(e)}",
                "raw_response": response.choices[0].message.content if 'response' in locals() else None
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": "llm_error",
                "error_message": str(e)
            }
    
    def _process_multiple_chunks(self, chunks: List[str], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process multiple transcription chunks and merge results"""
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            chunk_summary = self._process_single_chunk(chunk, context)
            
            if chunk_summary.get("status") == "error":
                return chunk_summary  # Return error immediately
            
            chunk_summaries.append(chunk_summary)
        
        # Merge chunk summaries
        return self._merge_chunk_summaries(chunk_summaries)
    
    def _merge_chunk_summaries(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple chunk summaries into a single comprehensive summary"""
        if not summaries:
            return {"status": "error", "error_message": "No summaries to merge"}
        
        # Take the first summary as base
        merged = summaries[0].copy()
        
        # Merge lists from other summaries
        for summary in summaries[1:]:
            # Merge discussion points
            if "key_discussion_points" in summary:
                merged["key_discussion_points"].extend(summary["key_discussion_points"])
            
            # Merge decisions
            if "decisions_made" in summary:
                merged["decisions_made"].extend(summary["decisions_made"])
            
            # Merge action items
            if "action_items" in summary:
                merged["action_items"].extend(summary["action_items"])
            
            # Merge open questions
            if "open_questions" in summary:
                merged["open_questions"].extend(summary["open_questions"])
            
            # Merge participants
            if "participants" in summary:
                merged["participants"].extend(summary["participants"])
            
            # Merge additional insights
            if "additional_insights" in summary:
                merged["additional_insights"].extend(summary["additional_insights"])
        
        # Remove duplicates
        merged["key_discussion_points"] = list(set(merged.get("key_discussion_points", [])))
        merged["decisions_made"] = list(set(merged.get("decisions_made", [])))
        merged["open_questions"] = list(set(merged.get("open_questions", [])))
        
        # Update processing info
        merged["processing_info"]["chunks_processed"] = len(summaries)
        merged["processing_info"]["merge_completed"] = datetime.now().isoformat()
        
        return merged
    
    def save_summary(self, summary: Dict[str, Any], filepath: str) -> bool:
        """
        Save the summary to a JSON file
        
        Args:
            summary: Summary dictionary
            filepath: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving summary: {e}")
            return False
    
    def format_summary_for_display(self, summary: Dict[str, Any]) -> str:
        """
        Format the summary for human-readable display
        
        Args:
            summary: Summary dictionary
            
        Returns:
            Formatted string representation
        """
        if summary.get("status") == "error":
            return f"Error: {summary.get('error_message', 'Unknown error')}"
        
        formatted = f"""
MEETING SUMMARY
===============

Title: {summary.get('meeting_title', 'Not specified')}
Date & Time: {summary.get('date_time', 'Not specified')}

PARTICIPANTS:
"""
        
        participants = summary.get('participants', [])
        if participants:
            for p in participants:
                formatted += f"- {p.get('name', 'Unknown')} ({p.get('role', 'Role not specified')})\n"
        else:
            formatted += "- Not specified\n"
        
        formatted += f"""
HIGH-LEVEL SUMMARY:
{summary.get('high_level_summary', 'Not available')}

KEY DISCUSSION POINTS:
"""
        
        for point in summary.get('key_discussion_points', []):
            formatted += f"• {point}\n"
        
        formatted += "\nDECISIONS MADE:\n"
        for decision in summary.get('decisions_made', []):
            formatted += f"• {decision}\n"
        
        formatted += "\nACTION ITEMS:\n"
        for item in summary.get('action_items', []):
            formatted += f"• {item.get('task', 'Unknown task')} "
            formatted += f"(Assignee: {item.get('assignee', 'Not specified')}, "
            formatted += f"Due: {item.get('due_date', 'Not specified')})\n"
        
        formatted += "\nOPEN QUESTIONS:\n"
        for question in summary.get('open_questions', []):
            formatted += f"• {question}\n"
        
        sentiment = summary.get('sentiment_analysis', {})
        formatted += f"""
MEETING ANALYSIS:
• Overall Tone: {sentiment.get('overall_tone', 'Not analyzed')}
• Engagement Level: {sentiment.get('engagement_level', 'Not analyzed')}
• Consensus Level: {sentiment.get('consensus_level', 'Not analyzed')}
"""
        
        return formatted

# Example usage function
# def example_usage():
#     """Example of how to use the MeetingTranscriptionSummarizer"""
    
#     # Initialize the summarizer (make sure to set your GROQ_API_KEY environment variable)
#     summarizer = MeetingTranscriptionSummarizer(
#         model="llama3-70b-8192",  # or "mixtral-8x7b-32768"
#         temperature=0.1
#     )
    
#     # Example transcription (this would come from your transcription system)
#     sample_transcription = """
#     John: Good morning everyone, welcome to our weekly product review meeting. Today is March 15th, 2024.
    
#     Sarah: Thanks John. As the product manager, I wanted to discuss the Q2 roadmap and some critical decisions we need to make.
    
#     Mike: I've been analyzing the user feedback from our beta release. We're seeing some concerns about the new authentication flow.
    
#     John: That's concerning. What specific issues are users reporting?
    
#     Mike: Main complaints are about the two-factor authentication being too complex. About 30% of users are dropping off at that step.
    
#     Sarah: We need to decide whether to simplify the 2FA or provide better user guidance. This affects our security posture though.
    
#     John: I think we should simplify it for now and revisit enhanced security in Q3. Mike, can you create a simplified flow by next Friday?
    
#     Mike: Absolutely, I'll have a prototype ready by Thursday for review.
    
#     Sarah: Good. Another item - we need to finalize the pricing strategy for the premium tier. Marketing wants this by end of month.
    
#     John: I'll take that action item. I'll coordinate with finance and have a proposal ready by March 28th.
    
#     Sarah: Perfect. Any other concerns or questions before we wrap up?
    
#     Mike: Just one question - are we still planning to integrate with the third-party analytics tool?
    
#     John: That's still under evaluation. I'll have an update next week.
    
#     Sarah: Alright, I think that covers everything. Thanks everyone for a productive meeting.
#     """
    
#     # Summarize the transcription
#     print("Processing transcription...")
#     summary = summarizer.summarize_transcription(sample_transcription)
    
#     if summary.get("status") == "error":
#         print(f"Error: {summary.get('error_message')}")
#     else:
#         # Display formatted summary
#         print(summarizer.format_summary_for_display(summary))
        
#         # Save to file
#         if summarizer.save_summary(summary, "meeting_summary.json"):
#             print("\nSummary saved to meeting_summary.json")
        
#         # Print raw JSON for inspection
#         print("\n" + "="*50)
#         print("RAW JSON SUMMARY:")
#         print(json.dumps(summary, indent=2))


# example_usage()