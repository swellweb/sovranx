#include "sovranx/server/http_types.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace sovranx::server {

namespace {

std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t");
    if (begin == std::string::npos) return {};
    const auto end = s.find_last_not_of(" \t");
    return s.substr(begin, end - begin + 1);
}

const char* status_reason(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 429: return "Too Many Requests";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default:  return "Unknown";
    }
}

}  // namespace

bool CaseInsensitiveLess::operator()(const std::string& a,
                                     const std::string& b) const {
    return std::lexicographical_compare(
        a.begin(), a.end(), b.begin(), b.end(), [](char x, char y) {
            return std::tolower(static_cast<unsigned char>(x)) <
                   std::tolower(static_cast<unsigned char>(y));
        });
}

std::optional<HttpRequest> parse_http_request(const std::string& data) {
    const auto headers_end = data.find("\r\n\r\n");
    if (headers_end == std::string::npos) return std::nullopt;

    // Request line: METHOD SP TARGET SP VERSION.
    const auto line_end = data.find("\r\n");
    const std::string line = data.substr(0, line_end);
    const auto sp1 = line.find(' ');
    if (sp1 == std::string::npos) return std::nullopt;
    const auto sp2 = line.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return std::nullopt;

    HttpRequest req;
    req.method = line.substr(0, sp1);
    req.target = line.substr(sp1 + 1, sp2 - sp1 - 1);
    if (req.method.empty() || req.target.empty() || req.target[0] != '/')
        return std::nullopt;

    // Strip the query string: routing only looks at the path.
    const auto q = req.target.find('?');
    if (q != std::string::npos) req.target.resize(q);

    // Headers.
    std::size_t pos = line_end + 2;
    while (pos < headers_end) {
        const auto eol = data.find("\r\n", pos);
        const std::string header_line = data.substr(pos, eol - pos);
        pos = eol + 2;
        const auto colon = header_line.find(':');
        if (colon == std::string::npos) continue;  // tolerate junk lines
        req.headers[trim(header_line.substr(0, colon))] =
            trim(header_line.substr(colon + 1));
    }

    // Body per Content-Length (the server guarantees the data is complete).
    const auto cl = req.headers.find("Content-Length");
    if (cl != req.headers.end()) {
        const auto length =
            static_cast<std::size_t>(std::strtoull(cl->second.c_str(),
                                                   nullptr, 10));
        const std::size_t body_start = headers_end + 4;
        if (data.size() - body_start < length) return std::nullopt;
        req.body = data.substr(body_start, length);
    }
    return req;
}

std::string serialize_http_response(const HttpResponse& response) {
    std::string out = "HTTP/1.1 " + std::to_string(response.status) + " " +
                      status_reason(response.status) + "\r\n";
    for (const auto& [name, value] : response.headers)
        out += name + ": " + value + "\r\n";
    if (response.headers.find("Content-Length") == response.headers.end())
        out += "Content-Length: " + std::to_string(response.body.size()) +
               "\r\n";
    out += "\r\n";
    out += response.body;
    return out;
}

}  // namespace sovranx::server
