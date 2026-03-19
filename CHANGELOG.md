# Changelog

All notable changes to this project will be documented in this file.

This changelog is automatically managed by release-please from commits merged into `main`.

## [0.7.0](https://github.com/dimidagd/transformers-knn-adapter/compare/v0.6.0...v0.7.0) (2026-03-19)


### Features

* **pipeline:** share embedding selection helpers ([6651bea](https://github.com/dimidagd/transformers-knn-adapter/commit/6651bea622dbb37512a24c99a156376f2b60d8df))
* **pipeline:** share embedding selection helpers ([84f7907](https://github.com/dimidagd/transformers-knn-adapter/commit/84f7907b6ae7ddb1d731b3a0367bb835ddd53bcd))

## [0.6.0](https://github.com/dimidagd/transformers-knn-adapter/compare/v0.5.0...v0.6.0) (2026-03-18)


### Features

* **knn:** use distance-weighted voting in callback ([c63c62e](https://github.com/dimidagd/transformers-knn-adapter/commit/c63c62e438cd919da78444e2f1463f9d5500f331))
* **pipeline:** add Gaussian directions plotting script ([6ff00d9](https://github.com/dimidagd/transformers-knn-adapter/commit/6ff00d9f4df60d989e8e0fee0c496486d3964dde))
* **pipeline:** add plotting script and weighted KNN eval ([b6ffd98](https://github.com/dimidagd/transformers-knn-adapter/commit/b6ffd989745ee4a87b3b74d6115b60d99b1356c1))


### Bug Fixes

* **pipeline:** escape CLI help percent and lazy HFDataset train ([#23](https://github.com/dimidagd/transformers-knn-adapter/issues/23)) ([2e76bb0](https://github.com/dimidagd/transformers-knn-adapter/commit/2e76bb0f4dac746803e5e1c7d42ad40a98d7a937))

## [0.5.0](https://github.com/dimidagd/transformers-knn-adapter/compare/v0.4.0...v0.5.0) (2026-03-12)


### Features

* **knn:** add robust embedding extraction and eval controls ([abc233c](https://github.com/dimidagd/transformers-knn-adapter/commit/abc233cbca5fec9082c6c82e091bdad8ed574761))
* **pipeline:** unify embeddings via HF image feature pipeline ([7e7747e](https://github.com/dimidagd/transformers-knn-adapter/commit/7e7747e513fea8f0ef1f5c04fca2bb6d0605bc5e))
* **pipeline:** unify embeddings via HF image pipeline ([02b558a](https://github.com/dimidagd/transformers-knn-adapter/commit/02b558ae58df7e7ed83edacdf30b7fadef5aca36))


### Documentation

* **docs:** add repo skills for CLI workflows ([b42a956](https://github.com/dimidagd/transformers-knn-adapter/commit/b42a95627f57e80a9d9f9b1fdcecdbd0ea70e9b9))

## [0.4.0](https://github.com/dimidagd/transformers-knn-adapter/compare/v0.3.0...v0.4.0) (2026-03-09)


### Features

* **pipeline:** add optional square padding for image inputs ([e86c627](https://github.com/dimidagd/transformers-knn-adapter/commit/e86c627c1b4818629b9da71caff870c087d3fd33))
* **pipeline:** add skip-channel-information preprocessing option ([3493e60](https://github.com/dimidagd/transformers-knn-adapter/commit/3493e601a24e8f575c812e28278c3555eea4478d))


### Bug Fixes

* **cli:** default pad-to-square to false when arg is absent ([288b11d](https://github.com/dimidagd/transformers-knn-adapter/commit/288b11d1fbf6a87435fb52f14aeeeecdecde7355))

## [0.3.0](https://github.com/dimidagd/transformers-knn-adapter/compare/v0.2.1...v0.3.0) (2026-03-06)


### Features

* **pipeline:** support local imagefolder dataset paths ([890bec4](https://github.com/dimidagd/transformers-knn-adapter/commit/890bec47b20f4b01f966dc4c101aacea9f8524c0))

## [0.2.1](https://github.com/dimidagd/transformers-knn-adapter/compare/v0.2.0...v0.2.1) (2026-03-05)


### Documentation

* **code:** add docstrings to CLI helper functions ([c3fdf4a](https://github.com/dimidagd/transformers-knn-adapter/commit/c3fdf4af184514bb33cfba5becbe6ecd0b045ba8))
* **readme:** add GitHub Actions badges for main workflows ([36ec158](https://github.com/dimidagd/transformers-knn-adapter/commit/36ec158e52fd77bbf9b9d6b5b6432fbee05b5639))

## [0.2.0](https://github.com/dimidagd/transformers-knn-adapter/compare/v0.1.0...v0.2.0) (2026-03-05)


### Features

* **cli:** add infer mode for single and batched image prediction ([47a291b](https://github.com/dimidagd/transformers-knn-adapter/commit/47a291b9156ad63d9263430366aded4687a8b16b))


### Documentation

* **readme:** add infer command usage example ([6b01105](https://github.com/dimidagd/transformers-knn-adapter/commit/6b01105379de1d4edc66294eaddd460c301352de))

## [Unreleased]
