from tqdm import tqdm
import torch
import concurrent.futures


def parallel_run(func, iter_items, args, num_workers=24):
    """
    Run a function in parallel on a list of items.
    
    Args:
        func (function): The function to run. The first argument should be the item to iterate over.
        args (list): List of additional arguments to pass to the function.
        iter_items (list): List of items to iterate over.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = [executor.submit(func, *item, *args) for item in iter_items]
        for _ in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
            pass

        
def get_walks_single(start, end, walk_len, context_sz, n_walks_per_node, walker):
    # Perform random walks starting from each node in `connected_nodes`
    trajectories = []
    for node in walker.connected_nodes[start:end]:
        for _ in range(n_walks_per_node):
            trajectory = walker.walk(node, walk_len)
            trajectories.append(trajectory)

    # Convert the walks into training samples
    walks = []
    for trajectory in trajectories:
        for cent in range(context_sz, walk_len - context_sz, 2):
            walks.append(trajectory[cent - context_sz : cent + context_sz + 1])
    walks = torch.tensor(walks)
    torch.save(walks, f'.tmp/walks_{start}_{end}.pth')