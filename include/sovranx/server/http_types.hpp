#pragma once

#include <map>
#include <optional>
#include <string>

namespace sovranx::server {

// Case-insensitive header map (HTTP header names are case-insensitive).
struct CaseInsensitiveLess {
    bool operator()(const std::string& a, const std::string& b) const;
};

using HeaderMap = std::map<std::string, std::string, CaseInsensitiveLess>;

struct HttpRequest {
    std::string method;  // uppercase: GET, POST, ...
    std::string target;  // path only, e.g. "/v1/completions"
    HeaderMap headers;
    std::string body;

    // Header value or empty string.
    std::string header(const std::string& name) const {
        const auto it = headers.find(name);
        return it == headers.end() ? "" : it->second;
    }
};

struct HttpResponse {
    int status = 200;
    HeaderMap headers;
    std::string body;

    // Header value or empty string.
    std::string header(const std::string& name) const {
        const auto it = headers.find(name);
        return it == headers.end() ? "" : it->second;
    }
};

// Sink for handler output. Non-streaming responses call begin+chunk+end
// once; SSE streaming calls chunk repeatedly.
class ResponseWriter {
public:
    virtual ~ResponseWriter() = default;
    virtual void begin(int status, const HeaderMap& headers) = 0;
    virtual void chunk(const std::string& data) = 0;
    virtual void end() = 0;
};

// Parses one full HTTP/1.1 request (request line + headers + body per
// Content-Length). Returns nullopt on malformed input. `data` must contain
// the complete request; incremental reads are the server's job.
std::optional<HttpRequest> parse_http_request(const std::string& data);

// Serializes a response with Content-Length and the standard line endings.
std::string serialize_http_response(const HttpResponse& response);

}  // namespace sovranx::server
