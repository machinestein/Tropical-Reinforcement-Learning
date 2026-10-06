#!/usr/bin/env bash
configure_gemma_webshop_runtime() {
    local gemma_java_home gemma_java_bin gemma_jvm_dir gemma_libjsig="" candidate
    local -a candidates=()
    if [[ -n ${JVM_PATH:-} ]]; then
        [[ -f $JVM_PATH ]] || { printf 'Gemma WebShop: JVM_PATH does not exist: %s\n' "$JVM_PATH" >&2; return 2; }
        gemma_jvm_dir=$(dirname -- "$(readlink -f -- "$JVM_PATH")")
        candidates=("$gemma_jvm_dir/libjsig.so" "$gemma_jvm_dir/../libjsig.so")
    else
        gemma_java_home=${JAVA_HOME:-${JDK_HOME:-${JRE_HOME:-}}}
        if [[ -z $gemma_java_home ]]; then
            gemma_java_bin=$(command -v javac || command -v java) || {
                printf 'Gemma WebShop: Java is missing; configure Java 21+ and JAVA_HOME.\n' >&2; return 2;
            }
            gemma_java_home=$(dirname -- "$(dirname -- "$(readlink -f -- "$gemma_java_bin")")")
        fi
        gemma_java_home=${gemma_java_home%/}
        gemma_java_home=${gemma_java_home%/bin}
        candidates=("$gemma_java_home/lib/libjsig.so" "$gemma_java_home/lib/server/libjsig.so")
    fi
    for candidate in "${candidates[@]}"; do
        if [[ -r $candidate ]]; then
            gemma_libjsig=$(readlink -f -- "$candidate")
            break
        fi
    done
    if [[ -z $gemma_libjsig ]]; then
        printf 'Gemma WebShop: libjsig.so is missing from the selected JVM. Use a full Java 21+ JDK and set JAVA_HOME (or JVM_PATH) to it.\n' >&2
        return 2
    fi
    if [[ $gemma_libjsig == *[[:space:]:]* ]]; then
        printf 'Gemma WebShop: LD_PRELOAD requires a JDK path without spaces or colons: %s\n' "$gemma_libjsig" >&2
        return 2
    fi
    local gemma_preloads=${LD_PRELOAD:-}
    gemma_preloads=${gemma_preloads//:/ }
    if [[ " $gemma_preloads " != *" $gemma_libjsig "* ]]; then
        export LD_PRELOAD="$gemma_libjsig${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
    printf 'Gemma WebShop JVM signal chaining: %s\n' "$gemma_libjsig"
}
