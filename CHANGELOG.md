# Changelog

## [0.1.1](https://github.com/nvandessel/sproink/compare/sproink-v0.1.0...sproink-v0.1.1) (2026-04-21)


### Bug Fixes

* address remaining code review gaps (14 issues) ([#31](https://github.com/nvandessel/sproink/issues/31)) ([52acea7](https://github.com/nvandessel/sproink/commit/52acea788e4b97b1b462ba7acc4ca4c059e94dff))
* audit round 2 — config validation, distance parity, FFI hardening ([#30](https://github.com/nvandessel/sproink/issues/30)) ([9bba357](https://github.com/nvandessel/sproink/commit/9bba35752314a78c70b08d3d66fa9072274d6645))
* complete final review gaps (I3, I16) and fix staticlib packaging ([#33](https://github.com/nvandessel/sproink/issues/33)) ([ebfb82b](https://github.com/nvandessel/sproink/commit/ebfb82bf71d44ef27de070a2e19abef8bd5f95df))
* complete sproink review — engine, tests, affinity, hebbian, FFI, CI/CD ([#28](https://github.com/nvandessel/sproink/issues/28)) ([a7cca87](https://github.com/nvandessel/sproink/commit/a7cca8743e15ff6b61b86952e607243e113d2732))
* **ffi:** eliminate UB, use u8 for bool param, reject invalid edge kinds ([#27](https://github.com/nvandessel/sproink/issues/27)) ([c8053b7](https://github.com/nvandessel/sproink/commit/c8053b70ab1443fcdd5471000eb2ce439218459d))
* return Result from CsrGraph::build with bounds and overflow validation ([#18](https://github.com/nvandessel/sproink/issues/18)) ([c8eabb8](https://github.com/nvandessel/sproink/commit/c8eabb8731edf5296c2f82d0d51e388e351741e9))
* **types:** enforce serde validation and canonicalize -0.0 ([#17](https://github.com/nvandessel/sproink/issues/17)) ([3a2d683](https://github.com/nvandessel/sproink/commit/3a2d683fb3c0e0facfce2e9bf90c10dbc42bbcd3))

## 0.1.0 (2026-04-07)


### Features

* floop algorithm parity + Rust code excellence ([#10](https://github.com/nvandessel/sproink/issues/10)) ([dfe9bf4](https://github.com/nvandessel/sproink/commit/dfe9bf44abe328ba2a22505e3782c7ec465e29ee))
* reimplement sproink from spec with full TDD ([c8c096d](https://github.com/nvandessel/sproink/commit/c8c096dd7dc09c1771eafec6a17ea2025a50807c))


### Bug Fixes

* add explicit branch filter and workflow_dispatch to CI ([ec2a35a](https://github.com/nvandessel/sproink/commit/ec2a35afc1959cb0a3f669b102c68ab31fcd60db))
* Apache 2.0 license, soften codecov failure, repo metadata ([5c56e58](https://github.com/nvandessel/sproink/commit/5c56e583b0dc984c9d26c6d550bd0b85efca763b))
* simplify CI matrix, remove workflow_dispatch ([5739952](https://github.com/nvandessel/sproink/commit/57399527b9b39ebf70c3c743a945585f25cc9f2a))


### Refactoring

* simplify and polish after code review ([fddc781](https://github.com/nvandessel/sproink/commit/fddc7816b7dfd41df4d57280444b75d12aa98dcb))
