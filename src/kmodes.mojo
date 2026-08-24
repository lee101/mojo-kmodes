"""Compute kernels for k-modes and k-prototypes clustering."""

from std.runtime import initialize_runtime
from std.sys import simd_width_of as simdwidthof

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime U16Ptr = UnsafePointer[UInt16, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime PARALLEL_MIN_WORK = 262144
comptime PARALLEL_TASKS = 32


@export("mkm_initialize_runtime")
def mkm_initialize_runtime() abi("C"):
    """Initialize Mojo when loaded as a shared library."""
    initialize_runtime()


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def fptr(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def u16ptr(addr: Int) -> U16Ptr:
    return U16Ptr(unsafe_from_address=addr)


def u32ptr(addr: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=addr)


@always_inline
def matching_distance_scalar(
    x: IPtr, center: IPtr, d: Int
) -> Float64:
    var distance = 0.0
    for j in range(d):
        if x[j] != center[j]:
            distance += 1.0
    return distance


def matching_distance(x: IPtr, center: IPtr, d: Int) -> Float64:
    comptime W = simdwidthof[DType.float64]()
    if d < 2 * W:
        return matching_distance_scalar(x, center, d)
    var vector_distance = SIMD[DType.int64, W](0)
    var j = 0
    while j + W <= d:
        var mismatches = x.load[width=W](j).ne(
            center.load[width=W](j)
        )
        var increments = mismatches.select(
            SIMD[DType.int64, W](1), SIMD[DType.int64, W](0)
        )
        vector_distance += increments
        j += W
    var distance = Float64(vector_distance.reduce_add())
    while j < d:
        if x[j] != center[j]:
            distance += 1.0
        j += 1
    return distance


def squared_distance(x: FPtr, center: FPtr, d: Int) -> Float64:
    var distance = 0.0
    for j in range(d):
        var delta = x[j] - center[j]
        distance += delta * delta
    return distance


def nearest_mode(x: IPtr, centers: IPtr, d: Int, k: Int) -> Int:
    var best_cluster = 0
    var best = matching_distance(x, centers, d)
    for cluster in range(1, k):
        var candidate = matching_distance(x, centers + cluster * d, d)
        if candidate < best:
            best = candidate
            best_cluster = cluster
    return best_cluster


def nearest_prototype(
    xnum: FPtr,
    xcat: IPtr,
    centers_num: FPtr,
    centers_cat: IPtr,
    dnum: Int,
    dcat: Int,
    k: Int,
    gamma: Float64,
) -> Int:
    var best_cluster = 0
    var best = (
        squared_distance(xnum, centers_num, dnum)
        + gamma * matching_distance_scalar(xcat, centers_cat, dcat)
    )
    for cluster in range(1, k):
        var candidate = (
            squared_distance(xnum, centers_num + cluster * dnum, dnum)
            + gamma
            * matching_distance_scalar(
                xcat, centers_cat + cluster * dcat, dcat
            )
        )
        if candidate < best:
            best = candidate
            best_cluster = cluster
    return best_cluster


def mode_cost(
    x: IPtr,
    centers: IPtr,
    weights: FPtr,
    labels: IPtr,
    n: Int,
    d: Int,
    k: Int,
) -> Float64:
    var cost = 0.0
    for row in range(n):
        var cluster = nearest_mode(x + row * d, centers, d, k)
        labels[row] = Int64(cluster)
        cost += weights[row] * matching_distance(
            x + row * d, centers + cluster * d, d
        )
    return cost


def prototype_cost(
    xnum: FPtr,
    xcat: IPtr,
    centers_num: FPtr,
    centers_cat: IPtr,
    weights: FPtr,
    labels: IPtr,
    n: Int,
    dnum: Int,
    dcat: Int,
    k: Int,
    gamma: Float64,
) -> Float64:
    var cost = 0.0
    for row in range(n):
        var cluster = nearest_prototype(
            xnum + row * dnum,
            xcat + row * dcat,
            centers_num,
            centers_cat,
            dnum,
            dcat,
            k,
            gamma,
        )
        labels[row] = Int64(cluster)
        cost += weights[row] * (
            squared_distance(
                xnum + row * dnum, centers_num + cluster * dnum, dnum
            )
            + gamma
            * matching_distance_scalar(
                xcat + row * dcat, centers_cat + cluster * dcat, dcat
            )
        )
    return cost


@no_inline
def matching_dissim_rows(
    a: Int, b: Int, dst: Int, n: Int, d: Int
):
    @parameter
    @always_inline
    def process_chunk(chunk: Int) capturing -> None:
        var start = chunk * n // PARALLEL_TASKS
        var stop = (chunk + 1) * n // PARALLEL_TASKS
        for row in range(start, stop):
            fptr(dst)[row] = matching_distance(
                iptr(a) + row * d, iptr(b), d
            )

    if n * d >= PARALLEL_MIN_WORK:
        for chunk in range(PARALLEL_TASKS):
            process_chunk(chunk)
    else:
        for row in range(n):
            fptr(dst)[row] = matching_distance(
                iptr(a) + row * d, iptr(b), d
            )


@no_inline
def mode_labels_addresses(
    x: Int,
    centers: Int,
    labels: Int,
    n: Int,
    d: Int,
    k: Int,
):
    @parameter
    @always_inline
    def assign_chunk(chunk: Int) capturing -> None:
        var start = chunk * n // PARALLEL_TASKS
        var stop = (chunk + 1) * n // PARALLEL_TASKS
        for row in range(start, stop):
            u16ptr(labels)[row] = UInt16(
                nearest_mode(
                    iptr(x) + row * d, iptr(centers), d, k
                )
            )

    if n * d * k >= PARALLEL_MIN_WORK:
        for chunk in range(PARALLEL_TASKS):
            assign_chunk(chunk)
    else:
        for row in range(n):
            u16ptr(labels)[row] = UInt16(
                nearest_mode(
                    iptr(x) + row * d, iptr(centers), d, k
                )
            )


def unicode_key_matches(
    data: U32Ptr,
    keys: U32Ptr,
    cell: Int,
    key: Int,
    input_width: Int,
    key_width: Int,
) -> Bool:
    var width = max(input_width, key_width)
    for char in range(width):
        var input_char = UInt32(0)
        if char < input_width:
            input_char = data[cell * input_width + char]
        var key_char = UInt32(0)
        if char < key_width:
            key_char = keys[key * key_width + char]
        if input_char != key_char:
            return False
    return True


@no_inline
def encode_unicode_addresses(
    data: Int,
    keys: Int,
    offsets: Int,
    codes: Int,
    dst: Int,
    n: Int,
    d: Int,
    input_width: Int,
    key_width: Int,
):
    @parameter
    @always_inline
    def encode_chunk(chunk: Int) capturing -> None:
        var start = chunk * n // PARALLEL_TASKS
        var stop = (chunk + 1) * n // PARALLEL_TASKS
        for row in range(start, stop):
            for column in range(d):
                var cell = row * d + column
                var encoded = Int64(-1)
                for key in range(
                    Int(iptr(offsets)[column]),
                    Int(iptr(offsets)[column + 1]),
                ):
                    if unicode_key_matches(
                        u32ptr(data),
                        u32ptr(keys),
                        cell,
                        key,
                        input_width,
                        key_width,
                    ):
                        encoded = iptr(codes)[key]
                        break
                iptr(dst)[cell] = encoded

    if n * d >= PARALLEL_MIN_WORK:
        for chunk in range(PARALLEL_TASKS):
            encode_chunk(chunk)
    else:
        encode_chunk(0)
        for chunk in range(1, PARALLEL_TASKS):
            encode_chunk(chunk)


def clear_mode_stats(
    counts: FPtr,
    cluster_sizes: IPtr,
    k: Int,
    total_categories: Int,
):
    for i in range(k * total_categories):
        counts[i] = 0.0
    for cluster in range(k):
        cluster_sizes[cluster] = 0


def update_modes(
    centers: IPtr,
    counts: FPtr,
    cluster_sizes: IPtr,
    offsets: IPtr,
    d: Int,
    k: Int,
    total_categories: Int,
):
    for cluster in range(k):
        if cluster_sizes[cluster] == 0:
            continue
        for attr in range(d):
            var start = Int(offsets[attr])
            var stop = Int(offsets[attr + 1])
            var best_value = 0
            var best_count = counts[cluster * total_categories + start]
            for value in range(1, stop - start):
                var candidate = counts[
                    cluster * total_categories + start + value
                ]
                if candidate > best_count:
                    best_count = candidate
                    best_value = value
            centers[cluster * d + attr] = Int64(best_value)


def initialize_modes(
    x: IPtr,
    centers: IPtr,
    weights: FPtr,
    membership: IPtr,
    counts: FPtr,
    cluster_sizes: IPtr,
    offsets: IPtr,
    n: Int,
    d: Int,
    k: Int,
    total_categories: Int,
):
    clear_mode_stats(counts, cluster_sizes, k, total_categories)
    for row in range(n):
        var cluster = nearest_mode(x + row * d, centers, d, k)
        membership[row] = Int64(cluster)
        cluster_sizes[cluster] += 1
        for attr in range(d):
            var value = Int(x[row * d + attr])
            var index = (
                cluster * total_categories + Int(offsets[attr]) + value
            )
            counts[index] += weights[row]
    update_modes(
        centers, counts, cluster_sizes, offsets, d, k, total_categories
    )


def move_mode_point(
    x: IPtr,
    centers: IPtr,
    move_weight: Float64,
    membership: IPtr,
    counts: FPtr,
    cluster_sizes: IPtr,
    offsets: IPtr,
    row: Int,
    old_cluster: Int,
    new_cluster: Int,
    d: Int,
    total_categories: Int,
):
    membership[row] = Int64(new_cluster)
    cluster_sizes[new_cluster] += 1
    cluster_sizes[old_cluster] -= 1
    for attr in range(d):
        var value = Int(x[row * d + attr])
        var start = Int(offsets[attr])
        var stop = Int(offsets[attr + 1])
        var new_index = new_cluster * total_categories + start + value
        var old_index = old_cluster * total_categories + start + value
        counts[new_index] += move_weight
        counts[old_index] -= move_weight

        var current = Int(centers[new_cluster * d + attr])
        if counts[new_cluster * total_categories + start + current] < counts[
            new_index
        ]:
            centers[new_cluster * d + attr] = Int64(value)

        if Int(centers[old_cluster * d + attr]) == value:
            var best_value = 0
            var best_count = counts[old_cluster * total_categories + start]
            for candidate_value in range(1, stop - start):
                var candidate_count = counts[
                    old_cluster * total_categories + start + candidate_value
                ]
                if candidate_count > best_count:
                    best_count = candidate_count
                    best_value = candidate_value
            centers[old_cluster * d + attr] = Int64(best_value)


def largest_cluster(cluster_sizes: IPtr, k: Int) -> Int:
    var largest = 0
    for cluster in range(1, k):
        if cluster_sizes[cluster] > cluster_sizes[largest]:
            largest = cluster
    return largest


def first_member(membership: IPtr, n: Int, cluster: Int) -> Int:
    for row in range(n):
        if Int(membership[row]) == cluster:
            return row
    return 0


def kmodes_fit(
    x: IPtr,
    centers: IPtr,
    weights: FPtr,
    membership: IPtr,
    labels: IPtr,
    counts: FPtr,
    cluster_sizes: IPtr,
    offsets: IPtr,
    epoch_costs: FPtr,
    n: Int,
    d: Int,
    k: Int,
    total_categories: Int,
    max_iter: Int,
) -> Int:
    initialize_modes(
        x,
        centers,
        weights,
        membership,
        counts,
        cluster_sizes,
        offsets,
        n,
        d,
        k,
        total_categories,
    )
    var cost = mode_cost(x, centers, weights, labels, n, d, k)
    epoch_costs[0] = cost
    var iterations = 0
    for iteration in range(max_iter):
        var moves = 0
        for row in range(n):
            var new_cluster = nearest_mode(x + row * d, centers, d, k)
            var old_cluster = Int(membership[row])
            if new_cluster == old_cluster:
                continue
            moves += 1
            move_mode_point(
                x,
                centers,
                weights[row],
                membership,
                counts,
                cluster_sizes,
                offsets,
                row,
                old_cluster,
                new_cluster,
                d,
                total_categories,
            )
            if cluster_sizes[old_cluster] == 0:
                var source = largest_cluster(cluster_sizes, k)
                var replacement = first_member(membership, n, source)
                move_mode_point(
                    x,
                    centers,
                    weights[replacement],
                    membership,
                    counts,
                    cluster_sizes,
                    offsets,
                    replacement,
                    source,
                    old_cluster,
                    d,
                    total_categories,
                )
        var new_cost = mode_cost(x, centers, weights, labels, n, d, k)
        iterations = iteration + 1
        epoch_costs[iterations] = new_cost
        if moves == 0 or new_cost >= cost:
            break
        cost = new_cost
    return iterations


def clear_prototype_stats(
    counts: FPtr,
    sums: FPtr,
    member_weights: FPtr,
    cluster_sizes: IPtr,
    k: Int,
    dnum: Int,
    total_categories: Int,
):
    clear_mode_stats(counts, cluster_sizes, k, total_categories)
    for i in range(k * dnum):
        sums[i] = 0.0
    for cluster in range(k):
        member_weights[cluster] = 0.0


def update_numeric_centers(
    centers_num: FPtr,
    sums: FPtr,
    member_weights: FPtr,
    k: Int,
    dnum: Int,
):
    for cluster in range(k):
        if member_weights[cluster] == 0.0:
            continue
        for attr in range(dnum):
            centers_num[cluster * dnum + attr] = (
                sums[cluster * dnum + attr] / member_weights[cluster]
            )


def initialize_prototypes(
    xnum: FPtr,
    xcat: IPtr,
    centers_num: FPtr,
    centers_cat: IPtr,
    weights: FPtr,
    membership: IPtr,
    counts: FPtr,
    sums: FPtr,
    member_weights: FPtr,
    cluster_sizes: IPtr,
    offsets: IPtr,
    n: Int,
    dnum: Int,
    dcat: Int,
    k: Int,
    total_categories: Int,
    gamma: Float64,
) -> Bool:
    clear_prototype_stats(
        counts,
        sums,
        member_weights,
        cluster_sizes,
        k,
        dnum,
        total_categories,
    )
    for row in range(n):
        var cluster = nearest_prototype(
            xnum + row * dnum,
            xcat + row * dcat,
            centers_num,
            centers_cat,
            dnum,
            dcat,
            k,
            gamma,
        )
        membership[row] = Int64(cluster)
        cluster_sizes[cluster] += 1
        member_weights[cluster] += weights[row]
        for attr in range(dnum):
            sums[cluster * dnum + attr] += (
                xnum[row * dnum + attr] * weights[row]
            )
        for attr in range(dcat):
            var value = Int(xcat[row * dcat + attr])
            counts[
                cluster * total_categories
                + Int(offsets[attr])
                + value
            ] += weights[row]
    for cluster in range(k):
        if cluster_sizes[cluster] == 0:
            return False
    update_numeric_centers(centers_num, sums, member_weights, k, dnum)
    update_modes(
        centers_cat,
        counts,
        cluster_sizes,
        offsets,
        dcat,
        k,
        total_categories,
    )
    return True


def update_numeric_after_move(
    xnum: FPtr,
    centers_num: FPtr,
    move_weight: Float64,
    sums: FPtr,
    member_weights: FPtr,
    row: Int,
    old_cluster: Int,
    new_cluster: Int,
    dnum: Int,
):
    member_weights[new_cluster] += move_weight
    member_weights[old_cluster] -= move_weight
    for attr in range(dnum):
        var weighted = xnum[row * dnum + attr] * move_weight
        sums[new_cluster * dnum + attr] += weighted
        sums[old_cluster * dnum + attr] -= weighted
        if member_weights[new_cluster] != 0.0:
            centers_num[new_cluster * dnum + attr] = (
                sums[new_cluster * dnum + attr]
                / member_weights[new_cluster]
            )
        else:
            centers_num[new_cluster * dnum + attr] = 0.0
        if member_weights[old_cluster] != 0.0:
            centers_num[old_cluster * dnum + attr] = (
                sums[old_cluster * dnum + attr]
                / member_weights[old_cluster]
            )
        else:
            centers_num[old_cluster * dnum + attr] = 0.0


def kprototypes_fit(
    xnum: FPtr,
    xcat: IPtr,
    centers_num: FPtr,
    centers_cat: IPtr,
    weights: FPtr,
    membership: IPtr,
    labels: IPtr,
    counts: FPtr,
    sums: FPtr,
    member_weights: FPtr,
    cluster_sizes: IPtr,
    offsets: IPtr,
    epoch_costs: FPtr,
    n: Int,
    dnum: Int,
    dcat: Int,
    k: Int,
    total_categories: Int,
    max_iter: Int,
    gamma: Float64,
) -> Int:
    if not initialize_prototypes(
        xnum,
        xcat,
        centers_num,
        centers_cat,
        weights,
        membership,
        counts,
        sums,
        member_weights,
        cluster_sizes,
        offsets,
        n,
        dnum,
        dcat,
        k,
        total_categories,
        gamma,
    ):
        return -1
    var cost = prototype_cost(
        xnum,
        xcat,
        centers_num,
        centers_cat,
        weights,
        labels,
        n,
        dnum,
        dcat,
        k,
        gamma,
    )
    epoch_costs[0] = cost
    var iterations = 0
    for iteration in range(max_iter):
        var moves = 0
        for row in range(n):
            var new_cluster = nearest_prototype(
                xnum + row * dnum,
                xcat + row * dcat,
                centers_num,
                centers_cat,
                dnum,
                dcat,
                k,
                gamma,
            )
            var old_cluster = Int(membership[row])
            if new_cluster == old_cluster:
                continue
            moves += 1
            update_numeric_after_move(
                xnum,
                centers_num,
                weights[row],
                sums,
                member_weights,
                row,
                old_cluster,
                new_cluster,
                dnum,
            )
            move_mode_point(
                xcat,
                centers_cat,
                weights[row],
                membership,
                counts,
                cluster_sizes,
                offsets,
                row,
                old_cluster,
                new_cluster,
                dcat,
                total_categories,
            )
            if cluster_sizes[old_cluster] == 0:
                var source = largest_cluster(cluster_sizes, k)
                var replacement = first_member(membership, n, source)
                update_numeric_after_move(
                    xnum,
                    centers_num,
                    weights[replacement],
                    sums,
                    member_weights,
                    replacement,
                    source,
                    old_cluster,
                    dnum,
                )
                move_mode_point(
                    xcat,
                    centers_cat,
                    weights[replacement],
                    membership,
                    counts,
                    cluster_sizes,
                    offsets,
                    replacement,
                    source,
                    old_cluster,
                    dcat,
                    total_categories,
                )
        var new_cost = prototype_cost(
            xnum,
            xcat,
            centers_num,
            centers_cat,
            weights,
            labels,
            n,
            dnum,
            dcat,
            k,
            gamma,
        )
        iterations = iteration + 1
        epoch_costs[iterations] = new_cost
        if moves == 0 or new_cost >= cost:
            break
        cost = new_cost
    return iterations


@export("mkm_matching_dissim")
def mkm_matching_dissim(
    a: Int, b: Int, dst: Int, n: Int, d: Int
) abi("C"):
    matching_dissim_rows(a, b, dst, n, d)


@export("mkm_mode_labels")
def mkm_mode_labels(
    x: Int,
    centers: Int,
    labels: Int,
    n: Int,
    d: Int,
    k: Int,
) abi("C"):
    mode_labels_addresses(x, centers, labels, n, d, k)


@export("mkm_encode_unicode")
def mkm_encode_unicode(
    data: Int,
    keys: Int,
    offsets: Int,
    codes: Int,
    dst: Int,
    n: Int,
    d: Int,
    input_width: Int,
    key_width: Int,
) abi("C"):
    encode_unicode_addresses(
        data,
        keys,
        offsets,
        codes,
        dst,
        n,
        d,
        input_width,
        key_width,
    )


@export("mkm_euclidean_dissim")
def mkm_euclidean_dissim(
    a: Int, b: Int, dst: Int, n: Int, d: Int
) abi("C"):
    var pa = fptr(a)
    var pb = fptr(b)
    var pdst = fptr(dst)
    for row in range(n):
        pdst[row] = squared_distance(pa + row * d, pb, d)


@export("mkm_mode_labels_cost")
def mkm_mode_labels_cost(
    x: Int,
    centers: Int,
    weights: Int,
    labels: Int,
    n: Int,
    d: Int,
    k: Int,
) abi("C") -> Float64:
    return mode_cost(
        iptr(x), iptr(centers), fptr(weights), iptr(labels), n, d, k
    )


@export("mkm_prototype_labels_cost")
def mkm_prototype_labels_cost(
    xnum: Int,
    xcat: Int,
    centers_num: Int,
    centers_cat: Int,
    weights: Int,
    labels: Int,
    n: Int,
    dnum: Int,
    dcat: Int,
    k: Int,
    gamma: Float64,
) abi("C") -> Float64:
    return prototype_cost(
        fptr(xnum),
        iptr(xcat),
        fptr(centers_num),
        iptr(centers_cat),
        fptr(weights),
        iptr(labels),
        n,
        dnum,
        dcat,
        k,
        gamma,
    )


@export("mkm_cao_init")
def mkm_cao_init(
    x: Int,
    centers: Int,
    density: Int,
    frequencies: Int,
    offsets: Int,
    n: Int,
    d: Int,
    k: Int,
    total_categories: Int,
) abi("C"):
    var px = iptr(x)
    var pc = iptr(centers)
    var pdensity = fptr(density)
    var pfreq = fptr(frequencies)
    var poffsets = iptr(offsets)
    for i in range(total_categories):
        pfreq[i] = 0.0
    for row in range(n):
        for attr in range(d):
            pfreq[
                Int(poffsets[attr]) + Int(px[row * d + attr])
            ] += 1.0
    var first = 0
    for row in range(n):
        var value = 0.0
        for attr in range(d):
            value += pfreq[
                Int(poffsets[attr]) + Int(px[row * d + attr])
            ] / Float64(n * d)
        pdensity[row] = value
        if value > pdensity[first]:
            first = row
    for attr in range(d):
        pc[attr] = px[first * d + attr]
    for cluster in range(1, k):
        var chosen = 0
        var chosen_score = -1.0
        for row in range(n):
            var min_distance = matching_distance(px + row * d, pc, d)
            for previous in range(1, cluster):
                var candidate = matching_distance(
                    px + row * d, pc + previous * d, d
                )
                if candidate < min_distance:
                    min_distance = candidate
            var score = min_distance * pdensity[row]
            if score > chosen_score:
                chosen_score = score
                chosen = row
        for attr in range(d):
            pc[cluster * d + attr] = px[chosen * d + attr]


@export("mkm_kmodes_fit")
def mkm_kmodes_fit(
    x: Int,
    centers: Int,
    weights: Int,
    membership: Int,
    labels: Int,
    counts: Int,
    cluster_sizes: Int,
    offsets: Int,
    epoch_costs: Int,
    n: Int,
    d: Int,
    k: Int,
    total_categories: Int,
    max_iter: Int,
) abi("C") -> Int:
    return kmodes_fit(
        iptr(x),
        iptr(centers),
        fptr(weights),
        iptr(membership),
        iptr(labels),
        fptr(counts),
        iptr(cluster_sizes),
        iptr(offsets),
        fptr(epoch_costs),
        n,
        d,
        k,
        total_categories,
        max_iter,
    )


@export("mkm_kprototypes_fit")
def mkm_kprototypes_fit(
    xnum: Int,
    xcat: Int,
    centers_num: Int,
    centers_cat: Int,
    weights: Int,
    membership: Int,
    labels: Int,
    counts: Int,
    sums: Int,
    member_weights: Int,
    cluster_sizes: Int,
    offsets: Int,
    epoch_costs: Int,
    n: Int,
    dnum: Int,
    dcat: Int,
    k: Int,
    total_categories: Int,
    max_iter: Int,
    gamma: Float64,
) abi("C") -> Int:
    return kprototypes_fit(
        fptr(xnum),
        iptr(xcat),
        fptr(centers_num),
        iptr(centers_cat),
        fptr(weights),
        iptr(membership),
        iptr(labels),
        fptr(counts),
        fptr(sums),
        fptr(member_weights),
        iptr(cluster_sizes),
        iptr(offsets),
        fptr(epoch_costs),
        n,
        dnum,
        dcat,
        k,
        total_categories,
        max_iter,
        gamma,
    )
