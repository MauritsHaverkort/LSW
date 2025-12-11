def give_line_width(indices, max_ranges, offset = 2, decrease_factor=0.5):
    assert len(indices) == len(max_ranges), "indices and max_ranges must have the same length"
    max_values_multiplied = [1]
    for i in range(len(max_ranges)):
        max_values_multiplied.append(max_values_multiplied[-1] * max_ranges[i])
    max_values_multiplied = np.array(max_values_multiplied)
    indices = np.array(indices)
    result = np.einsum('i,i->', indices, max_values_multiplied[:-1]) * decrease_factor
    return max_values_multiplied[-1]*decrease_factor - result + offset


def give_vector_terms_of_lattice_vectors(parameters:List["float"], lattice_vectors):
    result=0
    for i in range(len(parameters)):
        result+=parameters[i]*lattice_vectors[i]
    return result


def print_difference_to_timestamp(last_timestamp, debug_info, do_print):
    new_timestamp = int(time.time() * 1000)-last_timestamp
    print(str(new_timestamp)+"ms :"+str(debug_info))
    return new_timestamp
