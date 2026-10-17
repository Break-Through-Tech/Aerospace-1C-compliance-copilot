# Optional hosted API generator

`compliance_copilot/api_generator.py` is a generic client for an OpenAI-compatible Chat Completions endpoint. No paid provider is selected, and the client does not import a vendor SDK. The default demo backend remains local Qwen. A hosted request is possible only when you choose `--backend api` and provide the configuration below.

Local inference does not use this module. It does not need an API key, and it is not wrapped by the rate limiter.

## Request format

The client sends one non-streaming `POST` whose JSON body contains:

- `model`
- `messages`, in the order the caller supplied, with roles `system`, `user`, and `assistant`
- `max_tokens`
- `temperature`
- `stream: false`

It reads `choices[0].message.content` and returns that string. The audit pipeline still parses the string as JSON, checks the Pydantic schema, and builds citation quotes from retrieved clause records.

`COMPLIANCE_API_BASE_URL` is the service root or the full chat path. If the path already ends in `/chat/completions`, that URL is used. Otherwise the client appends `/chat/completions` once. Userinfo, query strings, and fragments are rejected. The default port is omitted from the stored endpoint identity.

Not every server that advertises OpenAI compatibility accepts `max_tokens` and `temperature`, or returns this exact response shape. A server that expects a different token-limit field, or that returns content in another property, will fail this client. Those failures are not rewritten into a successful audit.

## Configuration

Set these variables in the environment. Do not put them in source files, reports, or shell history that you share.

| Variable | Required | Purpose |
| --- | --- | --- |
| `COMPLIANCE_API_BASE_URL` | yes | HTTPS endpoint, unless local HTTP is explicitly allowed |
| `COMPLIANCE_API_KEY` | yes | Bearer token. Sent only on the `Authorization` header |
| `COMPLIANCE_API_MODEL` | yes, unless `--model` is set to something other than the local Qwen id | Model name sent to the provider |
| `COMPLIANCE_API_ALLOW_HTTP` | no | Set to `1` to permit an `http://` URL for a local server |

`--model` overrides `COMPLIANCE_API_MODEL` when it is not the local Qwen default. Leaving `--model` at that default does not send the Qwen id to the hosted API.

Demo flags for a hosted run:

- `--max-new-tokens` is sent as `max_tokens`. The default is 768.
- `--temperature` is sent as `temperature`. The default is 0. The local backend rejects any other value because Qwen decoding stays greedy.
- `--timeout` is the socket timeout in seconds. The default is 60. The local backend ignores it.
- `--max-requests` defaults to 5. It is the number of hosted attempts allowed during `--period-seconds`, which defaults to 60.
- `--max-attempts` defaults to 3 and includes the first attempt. The limiter retries; `APIGenerator` does not.

One agent requirement can use several hosted calls, and each retry counts. The defaults above can make a later requirement wait. Raise `--max-requests` to match the provider account. These flags do not change local Qwen. The provider's own quota can still be lower than the values you choose.

A missing variable, a non-HTTPS remote URL, or a URL that contains credentials raises `ValueError` before a request. The message names the missing setting and does not include the secret or the rejected URL.

```bash
python scripts/run_rag_demo.py --backend api --ids NASA-SR-001
python scripts/run_agent_demo.py --backend api --ids NASA-SR-001
```

`--backend` defaults to `local`. Local mode still loads `LocalGenerator` and does not read the API variables.

API mode does not construct `LocalGenerator` and does not ask Hugging Face for the chat model's revision. Retrieval still uses the local embedding index. Pass `--embedding-revision` when you want that revision pinned; API mode does not look it up.

## Retries and caching

`APIGenerator` does not retry. With `--backend api`, the demos use this order:

```text
caller → CachedTextGenerator → RateLimitedCallable → APIGenerator
```

`--cache` is still off by default. When it is on, the cache is outside the limiter:

- A hit returns stored text and does not call the API or take a quota slot.
- A miss calls through `RateLimitedCallable`.
- The demo limiter defaults to 5 requests per 60 seconds and 3 attempts. `--max-requests`, `--period-seconds`, and `--max-attempts` change that quota for `--backend api` only.
- HTTP 429, 500, 502, 503, and 504 raise `TransientAPIError` with `status_code` and `retry_after`. The limiter retries those, and `TimeoutError` and `ConnectionError`, using the rules in [API rate limiting](rate-limiting.md).
- HTTP 401 and 403 are not retried.
- A successful retry can be stored.
- An exception, malformed JSON, or missing assistant text is not stored.
- `APIResponseError` is the malformed-response signal. It has no HTTP status, so the limiter does not treat it as transient.

Hosted cache keys add the normalized endpoint and the temperature to the existing generation fields: schema version, backend `openai-compatible`, model, revision, `max_new_tokens`, `do_sample`, device `api`, and the ordered messages. The endpoint has no userinfo, query, or fragment. Two models or two endpoints do not share an entry. Local keys omit endpoint and temperature, so existing local entries stay valid. The schema version remains 1.

`--batch-size` on the agent demo still groups finished rows. It does not send a batch to the provider. The RAG demo does not gain a batch flag.

## Security

The key is read from the environment or a constructor argument and placed on the `Authorization` header. It is not written to the cache key, the cache file, demo reports, logs, or exception text. Provider error bodies are discarded. Reports record `api_endpoint` as the normalized URL only, plus the model name and temperature.

Use HTTPS for any machine you do not control. `COMPLIANCE_API_ALLOW_HTTP=1` is only for a local development server.

## Tests without a network call

The automated tests inject a fake transport or a fake `urlopen`. They do not use a live key.

```bash
py -3 -m pytest tests/test_api_generator.py tests/test_infrastructure_integration.py -q --tb=line -p no:cacheprovider
py -3 scripts/demo_infrastructure_integration.py
```

`scripts/demo_infrastructure_integration.py` runs the real agent, pipeline, JSON cache, batch runner, limiter, and API client against a fake transport. It shows one successful retry, one cache hit, and one exhausted 429. The cache and report stay in a temporary directory. The script does not use a provider key, the network, or local Qwen.

## Limitations

This is one Chat Completions shape, not a promise that every compatible provider behaves the same way. Streaming, async clients, and parallel requests are not implemented. The client does not translate provider-specific error JSON into audit verdicts. An exhausted retry remains an agent or pipeline error, not a Gap. There is no usage accounting beyond the in-process limiter.
