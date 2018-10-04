licenses(["notice"])


config_setting(
    name = "build_with_openblas",
    define_values = {
        "build_with_openblas": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "enable_openblas",
    define_values = {
        "enable_openblas": "true",
        "build_with_openblas": "true",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "openblas_headers",
    srcs = glob(["include/*"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "openblas_libs_linux",
    srcs = [
            "lib/libopenblas.so.0"
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "openblas_blob",
    visibility = ["//visibility:public"],
    deps = [
            ":openblas_headers",
            ":openblas_libs_linux",
    ],
)
