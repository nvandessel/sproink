# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a Vulnerability

**Please do not open a public issue for security vulnerabilities.**

Use [GitHub's private vulnerability reporting](https://github.com/nvandessel/sproink/security/advisories/new) to report security issues. This ensures the report stays confidential until a fix is available.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Impact assessment
- Suggested fix (if any)

### Response timeline

- **48 hours** -- acknowledgment of your report
- **1 week** -- initial assessment and severity classification
- **30 days** -- target for fix and coordinated disclosure

### Security scope

Sproink is an in-memory library. The primary security surface is:

- **FFI boundary** (`ffi.rs`) -- all `unsafe` code is confined here. Reports about memory safety issues in the FFI layer are especially valuable.
- **Input validation** -- malformed graph data, out-of-range parameters, integer overflow.

Issues in pure-Rust safe code (e.g., algorithmic correctness, performance) are generally not security vulnerabilities and can be reported as regular issues.

## Acknowledgments

We appreciate responsible disclosure and will credit reporters in release notes (unless you prefer anonymity).
