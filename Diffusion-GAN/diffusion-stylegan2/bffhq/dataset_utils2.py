import torch


def create_balanced_dataset2(
    train_dataset: torch.utils.data.Dataset,
    rho: float = None,
    align_count: int = None,
    conflict_count: int = None,
    max_total_per_class: int = None,  
):
    align_indices = train_dataset.align_indices
    conflict_indices = train_dataset.conflict_indices
    n_align = [len(idxs) for idxs in align_indices.values()]
    n_conf = [len(idxs) for idxs in conflict_indices.values()]
    n_cl = [n_align[i] + n_conf[i] for i in range(2)]

    if rho is not None:
        """
        obj_n_align = [int(n_cl[i] * rho) for i in range(2)]
        obj_n_conf = [n_cl[i] - obj_n_align[i] for i in range(2)]
        actual_align_count = min(min(n_align), min(obj_n_align))
        actual_conflict_count = min(min(n_conf), min(obj_n_conf))
        actual_rho = actual_align_count / (actual_align_count + actual_conflict_count)
        if actual_rho > rho:  # too much aligned samples
            conflict_count = actual_conflict_count
            align_count = int(conflict_count * rho / (1 - rho))
        elif actual_rho < rho: # too much conflicting samples
            align_count = actual_align_count
            conflict_count = int(align_count * (1 - rho) / rho)
        else: # just right
            align_count = actual_align_count
            conflict_count = actual_conflict_count
            
        # upper bound on cardinality 
        if max_total_per_class is not None:
            current_total = align_count + conflict_count
            if current_total > max_total_per_class:
                scale = max_total_per_class / current_total
                align_count = int(align_count * scale)
                conflict_count = max_total_per_class - align_count
        """
        
        actual_conflict_count = min(n_conf)  # anchor: take as many conflict as available (balanced across classes)
        desired_align_count = int(actual_conflict_count * rho / (1 - rho))
        actual_align_count = min(desired_align_count, min(n_align))  # clamp to available

        align_count = actual_align_count
        conflict_count = actual_conflict_count
        
        if max_total_per_class is not None:
            current_total = align_count + conflict_count
            if current_total > max_total_per_class:
                # ricalcola conflict in base al rho target, non tenerlo fisso
                conflict_count = int(max_total_per_class * (1 - rho))
                align_count = max_total_per_class - conflict_count
                if conflict_count > min(n_conf):
                    raise ValueError(f"conflict_count={conflict_count} exceeds available {min(n_conf)}")
        
        effective_rho = align_count / (align_count + conflict_count)
        print(f"Using rho={rho} to set align_count={align_count}, conflict={conflict_count} (per class)")
        print(f"Effective rho: {effective_rho}")

    else:
        if align_count > min(n_align):
            print(f"{align_count} exceeds available aligned samples.")
            align_count = min(n_align)
            print(f"Using {align_count} instead.")
        if conflict_count > min(n_conf):
            print(f"{conflict_count} exceeds available conflicting samples.")
            conflict_count = min(n_conf)
            print(f"Using {conflict_count} instead.")
        print(
            f"Using {align_count} aligned and {conflict_count} conflicting samples (per class)."
        )
        print(f"Effective rho: {align_count / (align_count + conflict_count)}")

    selected_indices = []
    for class_label in range(2):
        selected_align = torch.randperm(n_align[class_label])[:align_count]
        selected_indices.extend([align_indices[class_label][idx] for idx in selected_align])
        selected_conflict = torch.randperm(n_conf[class_label])[:conflict_count]
        selected_indices.extend([conflict_indices[class_label][i] for i in selected_conflict])

    if max_total_per_class is not None:
        return torch.utils.data.Subset(train_dataset, selected_indices[:2*max_total_per_class])
    else:
        return torch.utils.data.Subset(train_dataset, selected_indices)