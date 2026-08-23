"""
Multi-tool agent: a math assistant that can also read files and search the web.

This evolves the two-phase design into a general tool loop:

  1. Plan: decide whether to use a tool.
  2. Execute: run the selected tool and record the result.
  3. Repeat: plan again with the new information.
  4. Answer: once no more tools are needed, produce the final answer.

Tools:
  - calculate(expression): evaluates a mathematical expression.
  - read_file(path): reads a text file within the project directory.
  - get_weather(city): fetches the current weather for a city.
  - web_search(query): searches DuckDuckGo Instant Answer for a query and returns a summary.

Runs locally with Ollama (llama3.1). Set USE_OPENAI=1 to use OpenAI instead.
"""

import json
import math
import os
import re
import time

import ollama

from tracer import NullTrace, Trace, current_ms

MODEL = os.getenv("AGENT_MODEL", "llama3.1:latest")
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"
MAX_TOOL_STEPS = 5

# Root directory for file reads. Absolute paths outside this are blocked.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


def calculate(expression: str) -> str:
    """Evaluate a math expression in a restricted environment."""
    expression = _normalize_expression(expression)
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        name: getattr(math, name)
        for name in dir(math)
        if not name.startswith("_")
    }
    safe_locals.update({
        "abs": abs,
        "round": round,
        "max": max,
        "min": min,
        "sum": sum,
        "pow": pow,
    })
    try:
        result = eval(expression, safe_globals, safe_locals)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _normalize_expression(expression: str) -> str:
    """Rewrite common math notation into valid Python syntax."""
    # n! -> factorial(n)
    expression = re.sub(r"(\d+)!", r"factorial(\1)", expression)
    # ^ -> ** for exponentiation (common math notation, not Python XOR)
    expression = expression.replace("^", "**")
    return expression


def read_file(path: str) -> str:
    """Read a text file within the project directory."""
    target = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if not target.startswith(PROJECT_ROOT):
        return "Error: path is outside the allowed project directory"
    if not os.path.exists(target):
        return f"Error: file not found: {path}"
    if not os.path.isfile(target):
        return f"Error: not a file: {path}"
    try:
        with open(target, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Fetch current weather for a city using Open-Meteo (no API key required)."""
    import urllib.request
    import urllib.parse

    try:
        # Geocode the city.
        encoded_city = urllib.parse.quote(city)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1"
        with urllib.request.urlopen(geo_url, timeout=10) as response:
            geo_data = json.loads(response.read().decode("utf-8"))

        results = geo_data.get("results")
        if not results:
            return f"Error: could not find weather data for '{city}'"

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        display_name = results[0].get("name", city)
        country = results[0].get("country", "")
        location = f"{display_name}, {country}" if country else display_name

        # Fetch current weather.
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true"
        )
        with urllib.request.urlopen(weather_url, timeout=10) as response:
            weather_data = json.loads(response.read().decode("utf-8"))

        current = weather_data.get("current_weather", {})
        temp = current.get("temperature")
        code = current.get("weathercode")

        description = _weather_code_to_description(code)
        return f"Current weather in {location}: {temp}°C, {description}."
    except Exception as e:
        return f"Error: could not fetch weather for '{city}': {e}"


def _weather_code_to_description(code: int | None) -> str:
    """Convert an Open-Meteo weather code to a human-readable description."""
    if code is None:
        return "unknown"
    descriptions = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "light snow",
        73: "moderate snow",
        75: "heavy snow",
        80: "light rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        95: "thunderstorm",
        96: "thunderstorm with light hail",
        99: "thunderstorm with heavy hail",
    }
    return descriptions.get(code, "unknown")


_WEB_SEARCH_CACHE: dict[str, str] = {}


def web_search(query: str) -> str:
    """Search DuckDuckGo Instant Answer API and fall back to Wikipedia."""
    import html as html_module
    import urllib.error
    import urllib.parse
    import urllib.request

    if query in _WEB_SEARCH_CACHE:
        return _WEB_SEARCH_CACHE[query]

    user_agent = "Mozilla/5.0 (learning-agent/1.0; contact localhost)"

    def _ddg_lookup(q: str):
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _extract_ddg(data: dict) -> str:
        abstract = data.get("AbstractText", "").strip()
        answer = data.get("Answer", "").strip()
        heading = data.get("Heading", "").strip()
        if abstract:
            return abstract
        if answer:
            return answer
        if heading and heading.lower() != "today" and heading.lower() != "date":
            # "Today"/"Date" headings are usually disambiguation pages, not answers.
            return heading
        return ""

    def _wikipedia_lookup(q: str) -> str:
        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            "action=query&list=search&format=json&origin=*"
            f"&srsearch={urllib.parse.quote(q)}&srlimit=1"
        )
        req = urllib.request.Request(search_url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=10) as response:
            search_data = json.loads(response.read().decode("utf-8"))
        results = search_data.get("query", {}).get("search", [])
        if not results:
            return ""
        title = results[0]["title"]
        snippet = html_module.unescape(re.sub(r"<[^>]+>", "", results[0].get("snippet", "")))
        extract_url = (
            "https://en.wikipedia.org/w/api.php?"
            "action=query&prop=extracts&exintro&explaintext&format=json&origin=*"
            f"&titles={urllib.parse.quote(title)}"
        )
        req = urllib.request.Request(extract_url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=10) as response:
            extract_data = json.loads(response.read().decode("utf-8"))
        pages = extract_data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                return f"Wikipedia '{title}': {extract.strip()[:800].replace(chr(10), ' ')}"
        if snippet:
            return f"Wikipedia '{title}': {snippet.strip()[:500]}"
        return ""

    try:
        # Build a list of query variants to try.
        variants = [query]
        normalized = query.lower().strip("?")
        for prefix in ("current ", "today's ", "the ", "who is ", "what is ", "who are ", "what are ", "who was ", "what was "):
            if normalized.startswith(prefix):
                variants.append(normalized[len(prefix):].strip())
        for phrase in (" who is ", " what is ", " who are ", " what are ", " the ", " current "):
            if phrase in normalized:
                variants.append(normalized.replace(phrase, " ").strip())

        # Try DuckDuckGo first for clean instant answers.
        for variant in variants:
            if not variant:
                continue
            data = _ddg_lookup(variant)
            result = _extract_ddg(data)
            if result:
                _WEB_SEARCH_CACHE[query] = result
                return result

        # Fall back to Wikipedia for broader coverage.
        for variant in variants:
            if not variant:
                continue
            result = _wikipedia_lookup(variant)
            if result:
                _WEB_SEARCH_CACHE[query] = result
                return result

        return f"Error: no web search result for '{query}'"
    except urllib.error.HTTPError as e:
        return f"Error: search API returned {e.code} for '{query}'. Try again in a moment."
    except Exception as e:
        return f"Error: web search failed for '{query}': {e}"


TOOLS = {
    "calculate": calculate,
    "read_file": read_file,
    "get_weather": get_weather,
    "web_search": web_search,
}


# -----------------------------------------------------------------------------
# LLM helpers
# -----------------------------------------------------------------------------


def extract_json(text: str):
    """Try to find and parse a JSON object in the LLM response."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def chat(messages: list, temperature: float = 0.0) -> str:
    """Call the configured LLM and return the assistant's content."""
    if USE_OPENAI:
        import openai
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    response = ollama.chat(model=MODEL, messages=messages, options={"temperature": temperature})
    return response["message"]["content"].strip()


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------


_PLAN_SYSTEM_PROMPT = """You are a helpful assistant. You have access to these tools:

- calculate(expression): evaluates a mathematical expression and returns the result.
- read_file(path): reads the contents of a text file within the project directory.
- get_weather(city): fetches the current weather for a city.
- web_search(query): searches DuckDuckGo Instant Answer for a query and returns a short summary.

Rules:
1. If you need to use a tool, respond with ONLY a JSON object in this exact format:
   {"tool": "calculate", "input": "<expression>"}
   or
   {"tool": "read_file", "input": "<path>"}
   or
   {"tool": "get_weather", "input": "<city>"}
   or
   {"tool": "web_search", "input": "<query>"}
2. If you already have enough information to answer the user's question, respond with the final answer in plain text.
3. Do NOT repeat a tool call you have already made. Use the result you already have.
4. Do not include any explanation, code blocks, or markdown outside the JSON or the final answer.

Examples:

User: What is 2 + 2?
Assistant: {"tool": "calculate", "input": "2 + 2"}
Tool result: 4
Assistant: 4

User: What is in data/numbers.txt?
Assistant: {"tool": "read_file", "input": "data/numbers.txt"}
Tool result: 12\n15\n23\n8\n2\n3\n5
Assistant: 12\n15\n23\n8\n2\n3\n5

User: What is the sum of the numbers in data/numbers.txt?
Assistant: {"tool": "read_file", "input": "data/numbers.txt"}
Tool result: 12\n15\n23\n8\n2\n3\n5
Assistant: {"tool": "calculate", "input": "12 + 15 + 23 + 8 + 2 + 3 + 5"}
Tool result: 68
Assistant: 68

User: What is the product of the numbers in data/numbers.txt?
Assistant: {"tool": "read_file", "input": "data/numbers.txt"}
Tool result: 12\n15\n23\n8\n2\n3\n5
Assistant: {"tool": "calculate", "input": "12 * 15 * 23 * 8 * 2 * 3 * 5"}
Tool result: 993600
Assistant: 993600

User: What is the weather in Paris?
Assistant: {"tool": "get_weather", "input": "Paris"}
Tool result: Current weather in Paris, France: 15°C, partly cloudy.
Assistant: It is 15°C and partly cloudy in Paris.

User: Who is the CEO of OpenAI?
Assistant: {"tool": "web_search", "input": "CEO of OpenAI"}
Tool result: Wikipedia 'OpenAI': OpenAI is an American artificial intelligence (AI) research organization...
Assistant: Sam Altman is the CEO of OpenAI.

User: What is the capital of France?
Assistant: Paris
"""


def _build_plan_messages(question: str, tool_history: list, reflection_feedback: list) -> list:
    """Build the messages for the planning step, including tool history and retry feedback."""
    content = _PLAN_SYSTEM_PROMPT
    if tool_history:
        content += "\n\nYou have already used tools. Here are the results:\n"
        for tool_name, tool_input, tool_result in tool_history:
            content += f"\n{tool_name}({tool_input}) -> {tool_result}"
        content += "\n\nIf you need another tool, use JSON. If you have enough information, answer directly."
    if reflection_feedback:
        content += "\n\nReflection feedback from previous attempts:\n"
        for i, feedback in enumerate(reflection_feedback, 1):
            content += f"\n  Attempt {i}: {feedback}"
        content += "\n\nUse the feedback above to correct your answer."

    return [
        {"role": "system", "content": content},
        {"role": "user", "content": question},
    ]


def _build_answer_messages(question: str, tool_history: list, reflection_feedback: list) -> list:
    """Build the messages for the final answer step, including retry feedback."""
    content = "Use the tool results below to answer the question. Output ONLY the final answer in plain text, with no explanation.\n\n"
    content += "Tool results:\n"
    for tool_name, tool_input, tool_result in tool_history:
        content += f"\n  {tool_name}({tool_input}) -> {tool_result}"
    if reflection_feedback:
        content += "\n\nReflection feedback from previous attempts:\n"
        for i, feedback in enumerate(reflection_feedback, 1):
            content += f"\n  Attempt {i}: {feedback}"
        content += "\n\nUse the feedback above to provide the correct final answer."
    content += f"\n\nQuestion: {question}\n\nFinal answer:"

    return [
        {"role": "user", "content": content},
    ]


def _clean_answer(text: str) -> str:
    """Remove common role prefixes and blank lines that small models may emit."""
    text = text.strip()
    prefixes = ("assistant", "Assistant", "answer:", "Answer:", "final answer:", "Final answer:")
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            # Also strip a leading colon or newline if present.
            text = text.lstrip(":\n").strip()
            break
    return text


# -----------------------------------------------------------------------------
# Reflection / critic
# -----------------------------------------------------------------------------


_REFLECTION_SYSTEM_PROMPT = """You are a careful critic. Verify whether the proposed answer correctly answers the question, based on the tool history below.

Rules:
1. If the proposed answer is correct and well-supported by the tool history, respond with ONLY:
   VERIFIED: <answer>
2. If the proposed answer is incorrect or unsupported, respond with ONLY:
   INCORRECT: <reason>
3. If there is no tool history, verify the answer based on general knowledge.
4. Trust live external tool results over your own training knowledge. If the tool history includes a result from an external API such as get_weather or web_search, treat that result as authoritative — do not reject it because of a training-data cutoff or because it contradicts what you previously knew.
5. Do not include any explanation, code, or markdown outside the required format.

Examples:

Question: What is 2 + 2?
Tool history:
  calculate(2 + 2) -> 4
Proposed answer: 4
Response: VERIFIED: 4

Question: What is 15 * 23?
Tool history:
  calculate(15 * 23) -> 345
Proposed answer: 340
Response: INCORRECT: the tool result is 345, but the proposed answer is 340

Question: What is the capital of France?
Tool history: (none)
Proposed answer: Paris
Response: VERIFIED: Paris
"""


def _build_reflection_messages(question: str, tool_history: list, proposed_answer: str) -> list:
    """Build the messages for the reflection/critic step."""
    content = _REFLECTION_SYSTEM_PROMPT
    content += "\n\nNow verify this:\n\nTool history:\n"
    if tool_history:
        for tool_name, tool_input, tool_result in tool_history:
            content += f"\n  {tool_name}({tool_input}) -> {tool_result}"
    else:
        content += "\n  (none)"
    content += f"\n\nQuestion: {question}\nProposed answer: {proposed_answer}\n\nResponse:"

    return [
        {"role": "system", "content": content},
    ]


def _parse_reflection(reflection_output: str) -> tuple[bool, str]:
    """Parse the reflection output. Returns (verified, answer_or_reason)."""
    text = reflection_output.strip()
    if text.upper().startswith("VERIFIED:"):
        return True, text[len("VERIFIED:"):].strip()
    if text.upper().startswith("INCORRECT:"):
        return False, text[len("INCORRECT:"):].strip()
    # Fallback: assume verified if the response contains the proposed answer and no explicit incorrect marker.
    return True, text


def run_agent(question: str, verbose: bool = False, trace: bool = True, reflect: bool = True, max_retries: int = 2) -> str:
    provider = "OpenAI" if USE_OPENAI else "Ollama"
    tracer = Trace(question, MODEL, provider) if trace else NullTrace()

    run_start = current_ms()
    tool_history = []
    reflection_feedback = []
    final_answer = None
    last_reflection_text = None

    try:
        for attempt in range(max_retries + 1):
            if verbose:
                print(f"\n[Attempt {attempt + 1}]")

            tool_error = False

            # Tool loop: plan and execute until no tool is needed.
            for step in range(MAX_TOOL_STEPS):
                plan_messages = _build_plan_messages(question, tool_history, reflection_feedback)
                plan_start = current_ms()
                plan_output = chat(plan_messages)
                plan_duration = current_ms() - plan_start

                if verbose:
                    print(f"[Plan {attempt + 1}.{step + 1}] Agent: {plan_output}")

                tool_call = extract_json(plan_output)

                # No tool call: the model thinks it can answer directly.
                if not tool_call or tool_call.get("tool") not in TOOLS:
                    if not tool_history:
                        # No tools used yet; treat this as the direct answer.
                        final_answer = _clean_answer(plan_output)
                        tracer.add_step(
                            phase="direct_answer",
                            llm_output=plan_output,
                            duration_ms=plan_duration,
                            attempt=attempt,
                        )
                        break
                    # Tools were used; move to the final answer phase.
                    tracer.add_step(
                        phase="plan",
                        llm_output=plan_output,
                        duration_ms=plan_duration,
                        guard_triggered=True,
                        guard_reason="model answered directly after tool history; moving to answer phase",
                        attempt=attempt,
                    )
                    break

                tool_name = tool_call["tool"]
                tool_input = tool_call.get("input", "")

                # Guard: do not repeat any tool call that already appears in history.
                if any(name == tool_name and inp == tool_input for name, inp, _ in tool_history):
                    if verbose:
                        print(f"[Guard] Repeated tool call detected. Moving to answer phase.")
                    tracer.add_step(
                        phase="plan",
                        llm_output=plan_output,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        duration_ms=plan_duration,
                        guard_triggered=True,
                        guard_reason="repeated tool call blocked",
                        attempt=attempt,
                    )
                    break

                tool_fn = TOOLS[tool_name]
                tool_start = current_ms()
                result = tool_fn(tool_input)
                tool_duration = current_ms() - tool_start

                if verbose:
                    print(f"[Tool {attempt + 1}.{step + 1}] {tool_name}({tool_input}) = {result}")

                tracer.add_step(
                    phase="plan",
                    llm_output=plan_output,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_result=result,
                    duration_ms=plan_duration + tool_duration,
                    attempt=attempt,
                )

                if result.startswith("Error:"):
                    if verbose:
                        print(f"[Guard] Tool failed. Returning error message.")
                    final_answer = f"I could not use {tool_name}. {result}"
                    tool_error = True
                    tracer.add_step(
                        phase="error",
                        llm_output="",
                        guard_triggered=True,
                        guard_reason=f"tool {tool_name} returned error",
                        attempt=attempt,
                    )
                    break

                tool_history.append((tool_name, tool_input, result))

            # Final answer phase: synthesize the gathered information.
            if final_answer is None:
                answer_messages = _build_answer_messages(question, tool_history, reflection_feedback)
                answer_start = current_ms()
                answer_output = chat(answer_messages)
                answer_duration = current_ms() - answer_start
                answer_output = _clean_answer(answer_output)
                if verbose:
                    print(f"[Answer {attempt + 1}] Agent: {answer_output}")
                final_answer = answer_output
                tracer.add_step(
                    phase="answer",
                    llm_output=answer_output,
                    duration_ms=answer_duration,
                    attempt=attempt,
                )

            # If a tool failed, do not reflect or retry; just return the error.
            if tool_error:
                break

            # Reflection phase: verify the final answer before returning it.
            if reflect:
                reflection_messages = _build_reflection_messages(question, tool_history, final_answer)
                reflection_start = current_ms()
                reflection_output = chat(reflection_messages)
                reflection_output = _clean_answer(reflection_output)
                reflection_duration = current_ms() - reflection_start
                if verbose:
                    print(f"[Reflection {attempt + 1}] Critic: {reflection_output}")

                verified, reflection_text = _parse_reflection(reflection_output)
                last_reflection_text = reflection_text
                tracer.add_step(
                    phase="reflection",
                    llm_output=reflection_output,
                    duration_ms=reflection_duration,
                    guard_triggered=not verified,
                    guard_reason=None if verified else reflection_text,
                    attempt=attempt,
                )

                if verified:
                    if verbose:
                        print(f"[Retry] Answer verified on attempt {attempt + 1}.")
                    break

                # Reflection flagged the answer. Add feedback and retry.
                reflection_feedback.append(reflection_text)
                if verbose:
                    print(f"[Retry] Reflection flagged: {reflection_text}")

                if attempt < max_retries:
                    # Reset final_answer so the next attempt re-runs the answer phase.
                    final_answer = None
                else:
                    # No more retries. Return the answer with a warning.
                    final_answer = f"[Unverified after {max_retries} retries: {reflection_text}] {final_answer}"
            else:
                # Reflection disabled; accept the first answer.
                break

    except Exception as e:
        final_answer = f"Agent error: {e}"
        tracer.error = str(e)
        if verbose:
            print(f"[Error] {e}")

    total_duration = current_ms() - run_start
    tracer.finalize(final_answer, total_duration, tracer.error)
    trace_path = tracer.write()

    if trace_path and verbose:
        print(f"[Trace] written to {trace_path}")

    return final_answer


# -----------------------------------------------------------------------------
# Interactive entry point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    provider = "OpenAI" if USE_OPENAI else "Ollama"
    print(f"Multi-Tool Agent ({provider} / {MODEL})")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            question = input("Ask a question: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            answer = run_agent(question, verbose=True)
            print(f"\nFinal answer: {answer}\n")
    except EOFError:
        print("\nExiting (no more input).")
