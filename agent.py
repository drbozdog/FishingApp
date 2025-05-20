"""
OpenAI Agent Manager - Handles OpenAI API interactions and response management
"""

from __future__ import annotations
import os
import json
import inspect
from typing import Any, Dict, List, Tuple, Callable
from colorama import init, Fore, Style
from openai import AzureOpenAI
from dotenv import load_dotenv
from datetime import datetime
import pathlib

# Initialize colorama
init(autoreset=True)

class OpenAIManager:
    def __init__(self):
        load_dotenv()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-04-01-preview",
        )
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini")
        print(f"Using model: {self.model}")
        
        # Create logs directory if it doesn't exist
        self.logs_dir = pathlib.Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

    @staticmethod
    def generate_tool_schema(fn: Callable) -> Dict[str, Any]:
        """Generate OpenAI tool schema from a function."""
        sig = inspect.signature(fn)
        props, required = {}, []
        for n, p in sig.parameters.items():
            ptype = "number" if p.annotation in (int, float) else "string"
            props[n] = {"type": ptype}
            if p.default is inspect._empty:
                required.append(n)
        return {
            "type": "function",
            "name": fn.__name__,
            "description": fn.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }

    def generate_tools_spec(self, tool_funcs: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """Generate OpenAI tools specification from a dictionary of functions."""
        return [self.generate_tool_schema(f) for f in tool_funcs.values()]

    @staticmethod
    def say(txt, col=Fore.WHITE):
        print(col + txt + Style.RESET_ALL)

    def dump(self, resp):
        if not resp.output:
            return
        
        # Log all output items
        for idx, item in enumerate(resp.output, 1):
            if item.type == "reasoning" and hasattr(item, "summary"):
                self.say(f"\n[{idx}] REASONING SUMMARY:", Fore.BLUE)
                for summary in item.summary:
                    if summary.type == "summary_text":
                        self.say(f"  {summary.text}", Fore.YELLOW)
            elif item.type == "message":
                self.say(f"[{idx}] ASSISTANT MESSAGE ➜ {item.content[0].text}", Fore.MAGENTA)
            elif item.type == "function_call":
                self.say(f"[{idx}] TOOL CALL ➜ {item.name}({item.arguments})", Fore.YELLOW)

    def execute_tool_call(self, call: Any, tool_funcs: Dict[str, Callable]) -> Dict[str, Any]:
        """Execute a single tool call and return the result."""
        fn = tool_funcs[call.name]
        args = json.loads(call.arguments or "{}")
        try:
            result = fn(**args)
            self.say(f"Tool success ✓ Result: {json.dumps(result)[:400]}...", Fore.GREEN)
            return {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result)
            }
        except Exception as exc:
            result = {"error": str(exc)}
            self.say(f"Tool failed ✗ Error: {exc}", Fore.RED)
            return {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result)
            }

    def save_conversation(self, question: str, final_response: str, raw_responses: list, token_usage: dict) -> str:
        """Save the complete conversation details to a timestamped file in a readable format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.logs_dir / f"conversation_{timestamp}.log"
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Write header
            f.write("="*50 + "\n")
            f.write(f"Conversation Log - {timestamp}\n")
            f.write("="*50 + "\n\n")
            
            # Write initial question
            f.write(f"INITIAL QUESTION: {question}\n")
            f.write("-"*50 + "\n\n")
            
            # Write each response in sequence
            for response_idx, r in enumerate(raw_responses, 1):
                f.write(f"RESPONSE #{response_idx}\n")
                f.write("-"*30 + "\n")
                
                if not r.output:
                    f.write("No output in this response.\n\n")
                    continue
                
                # Log all output items
                for idx, item in enumerate(r.output, 1):
                    if item.type == "reasoning" and hasattr(item, "summary"):
                        f.write(f"\n[{idx}] REASONING SUMMARY:\n")
                        for summary in item.summary:
                            if summary.type == "summary_text":
                                f.write(f"  {summary.text}\n")
                    
                    elif item.type == "message":
                        f.write(f"[{idx}] ASSISTANT MESSAGE ➜ {item.content[0].text}\n")
                    
                    elif item.type == "function_call":
                        f.write(f"[{idx}] TOOL CALL ➜ {item.name}({item.arguments})\n")
                        
                    elif item.type == "function_call_output":
                        f.write(f"[{idx}] TOOL RESULT ➜ {item.output}\n")
                
                f.write("\n")
            
            # Write final response
            f.write("="*30 + " FINAL RESPONSE " + "="*30 + "\n")
            f.write(final_response)
            f.write("\n\n")
            
            # Write token usage statistics
            f.write("="*30 + " TOKEN USAGE " + "="*30 + "\n")
            f.write(f"MODEL: {token_usage['model']}\n")
            f.write(f"Prompt tokens:     {token_usage['prompt_tokens']:6d}\n")
            f.write(f"Completion tokens: {token_usage['completion_tokens']:6d}\n")
            f.write(f"Reasoning tokens:  {token_usage['reasoning_tokens']:6d}\n")
            f.write(f"Cached tokens:     {token_usage['cached_tokens']:6d}\n")
            f.write(f"Estimated cost:    ${token_usage['estimated_cost']:.4f}\n")
            
        self.say(f"\nConversation log saved to: {filename}", Fore.GREEN)
        return str(filename)

    def _serialize_output_item(self, item: Any) -> dict:
        """Helper method to serialize output items."""
        if item.type == "message":
            return {
                "type": "message",
                "content": [
                    {"type": c.type, "text": c.text}
                    for c in item.content
                ]
            }
        elif item.type == "function_call":
            return {
                "type": "function_call",
                "name": item.name,
                "arguments": item.arguments,
                "call_id": item.call_id
            }
        elif item.type == "function_call_output":
            return {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": item.output
            }
        return {"type": item.type, "data": str(item)}

    def tally(self, logs):
        prompt_total = 0
        completion_total = 0
        reasoning_total = 0
        cached_total = 0
        model = None

        for r in logs:
            if not r.usage:
                continue

            # Track the model being used (use the last one seen)
            if r.model:
                model = r.model

            prompt_total += r.usage.input_tokens
            completion_total += r.usage.output_tokens

            # Track reasoning tokens
            comp_details = getattr(r.usage, "output_tokens_details", None)
            if comp_details and getattr(comp_details, "reasoning_tokens", None):
                reasoning_total += comp_details.reasoning_tokens

            # Track cached tokens
            prompt_details = getattr(r.usage, "input_tokens_details", None)
            if prompt_details and getattr(prompt_details, "cached_tokens", None):
                cached_total += prompt_details.cached_tokens

        self.say("\nTOTAL TOKENS:", Fore.GREEN)
        self.say(f"  Prompt:     {prompt_total:6d}", Fore.CYAN)
        self.say(f"  Completion: {completion_total:6d}", Fore.CYAN)
        self.say(f"  Reasoning:  {reasoning_total:6d}", Fore.CYAN)
        self.say(f"  Cached:     {cached_total:6d}", Fore.CYAN)
        
        # Calculate cost based on model
        if model == "o3":
            # o3 pricing: $10/1M input, $2.50/1M cached, $40/1M output
            cost = (
                (prompt_total - cached_total) * 10.0/1_000_000 +  # non-cached input
                cached_total * 2.50/1_000_000 +                   # cached input
                completion_total * 40.0/1_000_000                 # output
            )
            self.say(f"\nMODEL: o3 (high-performance)", Fore.BLUE)
        elif model == "o4-mini":  # o4-mini or default
            # o4-mini pricing: $1.10/1M input, $0.275/1M cached, $4.40/1M output
            cost = (
                (prompt_total - cached_total) * 1.10/1_000_000 +  # non-cached input
                cached_total * 0.275/1_000_000 +                  # cached input
                completion_total * 4.40/1_000_000                 # output
            )
            self.say(f"\nMODEL: o4-mini (efficient)", Fore.BLUE)
        else:
            self.say(f"\nMODEL: unknown", Fore.BLUE)
            cost = 0

        self.say(f"EST. COST: ${cost:.4f}", Fore.RED)
        
        return {
            "model": model,
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
            "reasoning_tokens": reasoning_total,
            "cached_tokens": cached_total,
            "estimated_cost": cost
        }

    def run_conversation(self, question: str, system_msg: str, tool_funcs: Dict[str, Callable]) -> Tuple[str, list]:
        self.say("\n" + "="*50, Fore.BLUE)
        self.say("Starting new conversation", Fore.BLUE)
        self.say(f"Question: {question}", Fore.WHITE)
        self.say("="*50 + "\n", Fore.BLUE)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ]
        raw: list = []
        last_response_id = None
        tools_spec = self.generate_tools_spec(tool_funcs)

        while True:
            self.say("\nSending request to AI...", Fore.BLUE)
            resp = self.client.responses.create(
                model=self.model,
                tools=tools_spec,
                input=messages,
                reasoning={
                    "effort": "low",
                    "summary": "auto"
                }
            )
            raw.append(resp)
            self.dump(resp)
            last_response_id = resp.id

            # Add all response items to messages to maintain context
            messages.extend(resp.output)

            # Separate the items we care about
            tool_calls = [o for o in resp.output if o.type == "function_call"]
            assistant_msgs = [o for o in resp.output if o.type == "message"]

            self.say(f"\nFound {len(tool_calls)} tool calls and {len(assistant_msgs)} messages", Fore.BLUE)

            # If there are tool calls, execute them and continue the conversation
            if tool_calls:
                # Execute first tool call and add result to messages
                result = self.execute_tool_call(tool_calls[0], tool_funcs)
                messages.append(result)
                continue

            # If we got here, we have a final text response
            if assistant_msgs:
                text_chunks = [
                    chunk.text
                    for m in assistant_msgs
                    for chunk in m.content
                    if chunk.type == "output_text"
                ]
                final_text = "\n".join(text_chunks)
                self.say("\nConversation complete!", Fore.GREEN)
                
                # Get token usage
                token_usage = self.tally(raw)
                
                # Save the conversation
                self.save_conversation(question, final_text, raw, token_usage)
                
                return final_text, raw 

            raise RuntimeError("No message or function call in output") 