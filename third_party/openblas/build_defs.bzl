

def if_openblas(if_true, if_false = []):
    return select({
        str(Label("//third_party/openblas:build_with_openblas")): if_true,
        "//conditions:default": if_false,
    })

def if_enable_openblas(if_true, if_false = []):
    return select({
        str(Label("//third_party/openblas:enable_openblas")): if_true,
        "//conditions:default": if_false,
    })

def openblas_deps():
    return select({
        str(Label("//third_party/openblas:build_with_openblas")): ["//third_party/openblas:openblas_blob"],
        "//conditions:default": [],
    })