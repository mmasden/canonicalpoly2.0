import os
import torch
from torch import nn 
from torch.utils.data import DataLoader

import itertools 
import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super(NeuralNetwork, self).__init__()
        
        self.architecture=architecture
        
        self.flatten = nn.Flatten()
        self.linear_0 = nn.Linear(architecture[0],architecture[1])
        self.relu_0 = nn.ReLU()
        self.linear_1 = nn.Linear(architecture[1],architecture[2])


    def forward(self, x):
        x = self.flatten(x)
        
        self.activity_0 = self.linear_0(x) 
        self.activity_1 = self.linear_1(self.relu_0(self.activity_0))

        return self.activity_0, self.activity_1
    

class DeepNeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super(DeepNeuralNetwork, self).__init__()
        
        self.architecture=architecture
        
        self.flatten = nn.Flatten()
        self.linear_0 = nn.Linear(architecture[0],architecture[1])
        self.relu_0 = nn.ReLU()
        self.linear_1 = nn.Linear(architecture[1],architecture[2])
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(architecture[2],architecture[3])


    def forward(self, x):
        x = self.flatten(x)
        
        self.activity_0 = self.linear_0(x) 
        self.activity_1 = self.linear_1(self.relu_0(self.activity_0))
        self.activity_2 = self.linear_2(self.relu_1(self.activity_1))

        return self.activity_0, self.activity_1, self.activity_2 #, self.activity_3


class DeeperNeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super(DeeperNeuralNetwork, self).__init__()
        
        self.architecture=architecture
        
        self.flatten = nn.Flatten()
        self.linear_0 = nn.Linear(architecture[0],architecture[1])
        self.relu_0 = nn.ReLU()
        self.linear_1 = nn.Linear(architecture[1],architecture[2])
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(architecture[2],architecture[3])
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(architecture[3],architecture[4])


    def forward(self, x):
        x = self.flatten(x)
        
        self.activity_0 = self.linear_0(x) 
        self.activity_1 = self.linear_1(self.relu_0(self.activity_0))
        self.activity_2 = self.linear_2(self.relu_1(self.activity_1))
        self.activity_3 = self.linear_3(self.relu_2(self.activity_2))

        return self.activity_0, self.activity_1, self.activity_2, self.activity_3


#multiply two sign sequences
def multiply(v1, v2): 
    '''Computes the product of two sign sequences''' 

    if len(v1)!= len(v2): 
        raise ValueError("The two sign sequences must have same length")
        return
    else: 
                     
        # apply product definition 

        product = list(v1) 
        for i in range(len(product)): 
            if product[i]==0: 
                product[i] = v2[i]
       
        return tuple(product)
    
def multiply_torch(v1, v2):
    '''Computes the product of two sign sequences given as Torch tensors''' 

    if not torch.is_tensor(v1): 
        v1 = torch.tensor(v1) 
    if not torch.is_tensor(v2): 
        v2=torch.tensor(v2)

    product = v1.clone()
    locs = torch.where(product==0)[0]
    product[locs]=v2[locs]
    
    return product
    
def edge_connected(v1, v2,  dim = 2): 
    '''Checks if two vertices, in given ambient dimension, are connected by an edge.
    This occurs if the product commutes and the resulting cell is an edge'''
     
    p1 = multiply(v1,v2)
    p2 = multiply(v2,v1) 
    #print(p1,p2)
    if p1==p2 and sum([p==0 for p in p1]) == dim-1: 
        return True 
    else: 
        return False 

def edge_connected_torch(v1,v2,dim=2):     
    '''Checks if two vertices, in given ambient dimension, are connected by an edge.
    This occurs if the product commutes and the resulting cell is an edge'''

    p1 = multiply_torch(v1,v2)
    p2 = multiply_torch(v2,v1) 
    #print(p1,p2)
    if torch.equal(p1,p2) and torch.sum(p1==0) == dim-1: 
        return True 
    else: 
        return False 
    
    
def is_face(v, F): 
    '''Checks if the cell represented by v is a face of the cell represented by F'''
    p = multiply(v,F)

    if p == tuple(F): 
        return True 
    else: 
        return False 
    
def is_face_torch(v, F): 
    '''Checks if the cell represented by v is a face of the cell represented by F'''

    p = multiply_torch(v,F)

    if torch.equal(p,F): 
        return True 
    else: 
        return False   
    
def make_affine(matrix, bias, device='cpu'):
    A = torch.hstack([matrix, bias.reshape(len(matrix),1)])
    A = torch.vstack([A,torch.zeros(1,A.shape[1],device=device)])
    A[-1,-1] = 1 
    return A 


def make_linear(affine_matrix): 
    matrix = affine_matrix[0:-1,0:-1]
    bias = affine_matrix[:-1,-1]
    return matrix,bias



    
#obtain affine maps for each regifdon 
def get_layer_map_on_region(ss,weights,biases,device='cpu'): 
    ''' 
    Inputs sign sequence in layer and parameter list for layer. Returns map on region of that layer.
    '''
    
    base_A = make_affine(weights,biases, device=device)
    region_indices = torch.where(ss==-1)[0]
    #print(region_indices)
    r_map = torch.clone(base_A)
    r_map[region_indices,:] = 0

    return r_map 

def tensor_tuple_to_numpy(tt): 
    
    tt = np.array([t.cpu().detach().numpy() for t in tt])
    
    return tuple(tt)

def numpyize_plot_dict(pd): 
    
    
    pd2 = {tensor_tuple_to_numpy(tt):pd[tt].cpu().detach().numpy() for tt in pd}
    
    return pd2
    




def get_signs(dim):
     #[[1,1],[-1,-1],[1,-1],[-1,1]]      
    if dim == 1: 
        return [[1],[-1]]
    elif dim > 1: 
        signs = [] 
        for signlist in get_signs(dim-1):
            signs.append(signlist+[1]) 
            signs.append(signlist+[-1]) 
        
        return signs
    
    elif dim <1: 
        print("not valid")
        return


    
def get_ssr(ssv, ss_length, signs): 
    #record the existing vertex ss as np arrays or similar
    ssv_np = [ss[0:ss_length] for ss in ssv]
    
    #record the regions that are present as a set 
    ssr = []

    #loop through vertices and obtain regions which are adjacent
    for ss in ssv_np: 
        #ss_np = np.array(ss)
        locs = torch.where(ss==0)[0]
        dimension=len(locs)
        
        tempss=torch.clone(ss)
        #print(tempss)
        #print(ss)

        for sign in signs: 
            #print(sign)
            #print(tempss[locs])
            tempss[locs]=sign
            #print(tempss)
            
            ssr.append(tempss.clone())
                
                #print(ssr)
    
    ssr=torch.vstack(ssr)
    
    return torch.unique(ssr, dim=0)
    
    
def determine_existing_points(points, combos, model, region_ss=None, device='cpu'):
    ''' evaluates sign sequence of points matches existing sign sequence in region
    Region sign sequence should be truncated.'''
    
    image = model(points)

    # obtains sign sequence of initial vertices
    ssv = torch.hstack([torch.sign(image[i]) for i in range(len(image))])
    

    #force correct signs at intersections 
    for i in range(len(ssv)): 
        ssv[i,combos[i]]=0

    true_points = []
    true_ssv = []

    #determine if it is a face 
    
    region_len = 0 if region_ss is None else len(region_ss) 
    
    for  temp_pt, temp_ss in zip(points, ssv): 
        if region_ss is None or is_face_torch(temp_ss[0:region_len],region_ss): 
            true_points.append(temp_pt) #.cpu().detach().numpy()) 
            true_ssv.append(temp_ss) 
    
    if len(true_ssv)>1:
        true_ssv=torch.vstack(true_ssv)
        true_points=torch.vstack(true_points)
    
    
    
    return true_points, true_ssv
    
    
def get_all_maps_on_region(ss, depth, param_list, architecture, device='cpu'): 
       
    cumulative_architecture = [np.sum(architecture[1:i],dtype='int') for i in range(1,len(architecture)+1)]
    
    region_maps = []
    
    for i in range(depth): 
        layer_ss = ss[cumulative_architecture[i]:cumulative_architecture[i+1]]
        region_map_on_layer = get_layer_map_on_region(layer_ss,param_list[2*i],param_list[2*i+1], device=device)
        region_maps.append(region_map_on_layer)
        
        
    early_layer_maps = [region_maps[0]] 
    
    
    for rmap in region_maps[1:]:
        early_layer_maps.append(rmap @ early_layer_maps[-1])
    
        
    affine_layer_maps = [make_affine(param_list[2*i],param_list[2*i+1], device=device) for i in range(depth+1)]
    
    actual_layer_maps = [affine_layer_maps[0].detach()]
    
    for rmap, amap in zip(early_layer_maps, affine_layer_maps[1:]): 
        actual_layer_maps.append((amap@rmap).detach())
        
    return actual_layer_maps


def get_face_combos(ssr, existing_vertices): 
    '''Obtains minimal sets of hyperplanes forming the faces of a polyhedral region, 
       given a list of the sign sequences of its vertices '''

    combos=[]
    true_vertex_ssvs = [] 
    
    in_dim = sum(existing_vertices[0]==0)
    
    #get list of vertices which are a face of region with given ssr 
    
    for vertex in existing_vertices: 
        
        #truncate vertex sign sequence to appropriate length for layer
        
        if is_face_torch(vertex[0:len(ssr)],ssr): 
            true_vertex_ssvs.append(vertex[0:len(ssr)])
         
    true_vertex_ssvs = torch.vstack(true_vertex_ssvs)
    
    # all faces of C are the vertices of C with a number of their zeros
    # replaced by the corresponding entry of C 
    
    # the bent hyperplanes which intersect to form the faces of C are given 
    # by all subsets of the hyperplanes which intersect to form the vertices of C 
    # e.g. if BH 1n2n3 is an vertex of C, then 1n2 intersect to form a face, 2n3, 1n3 and
    # 1, 2, and 3 individually 
    
    # e.g. simplicial closure 
    
    # This should be shellable, so there should be a way to do this
    # which is much faster than below. 
    
    top_simplices = [torch.where(vertex==0)[0] for vertex in true_vertex_ssvs]
    top_simplices = torch.vstack(top_simplices)
    
    hyperplane_combos = [[]] 

    #loop through size of hyperplane combo 
    
    for i in range(1, in_dim+1): 

        #get all i-subsets of top simplices 

        hyperplane_combos.append([])
        
        combinations = torch.combinations(torch.arange(in_dim),r=i) 
        
        
        temp_h_combos = torch.vstack([top_simplices[:,combination] for combination in combinations])
        
        temp_h_combos = torch.unique(temp_h_combos,dim=0) 
        
        hyperplane_combos[i]=temp_h_combos                                    
        
    return hyperplane_combos

def range_exclude(n,exclusion): 
    for i in range(n): 
        if i not in exclusion: 
            yield(i)


#find possible intersections of bent hyperplanes, given a list of all neurons in earlier layers 
# and a list of neurons in later layers.

def find_intersections(in_dim, last_layer, last_biases, image_dim, ssr, architecture,hyperplane_combos=None, early_layer_maps=None, early_layer_biases=None, device='cpu'): 
    '''Given a polyhedral region R, in input space, layer_maps is a tensor of the activity functions
    of each neuron on the interior of that region. If this is the first layer, input None. 
    last_layer is the single layer after layer_maps which provides "new" bent hyperplanes.
    Returns the locations of the vertices, and which pairs of bent hyperplanes intersect 
    at those points.
    
    Returns: locations of points which represent possible vertices, the pairs of hyperplanes which 
    intersect to make those points'''
        
    last_layer = last_layer.detach()
    last_biases = last_biases.detach() 
    
    #If the last layer is the only layer then layer_maps is None. 
    if early_layer_maps is None or early_layer_biases is None: 
        
        #get at tensorfied list of all neurons in the output of the given layer 
        n_out=torch.arange(len(last_layer)) 
        
        #obtain all in_dim-combinations of the first layer's hyperplanes  
        
        if in_dim < (len(last_layer)/2+1): 
            combos = torch.combinations( n_out, r=in_dim)
        else: 
            complement_combos = torch.combinations(n_out, r=len(last_layer)-in_dim)

            combos = []

            for entry in complement_combos: 
                new_values = torch.tensor(list(range_exclude(len(last_layer),entry)))
                combos.append(new_values)

            combos = torch.vstack(combos)


            
        #solves for points

        points = torch.linalg.solve(last_layer[combos].detach(), -last_biases[combos].detach()) 
        
        return points, combos
    
    else:
        all_points = []
        combos = []

        n_between = torch.tensor(range(len(early_layer_biases)))
        n_out = torch.tensor(range(len(last_biases)))
        
        # loop through k, the number of new bent hyperplanes involved in intersection 
        # The number of new bent hyperplanes involved in the intersection 
        # is bounded above by the dimension of the image of the region in this layer! 
        
        # omit 0 because must include some from new layers; can't go above n_out
        for k in range(1,min(image_dim+1,in_dim,len(last_biases)+1)):  
            
            last_combos = torch.combinations(n_out, r=k) 
                      
            early_combos = hyperplane_combos[in_dim-k]
            
            old_vals=len(early_layer_maps)
            
            # worry about degeneracy only if it has been collapsed 
            
            if image_dim < in_dim:
                
                if image_dim ==0: 
                    pass
                else: 
                    
                    
                    # IF HYPERPLANES NONGENERIC SKIP 
                    # This occurs if image_dim < in_dim (the region has been collapsed) 
                    # and the bent hyperplanes from earlier layers intersect in a region 
                    # sent to too few dimensions to generically intersect with the. 
                    # next layer's hyperplanes.
                    # The latter occurs when, if taking the sign sequence of the region
                    # and setting all the BH's coordinates to 0, you have fewer 1's left than 
                    # the dimension minus the number of new hyperplanes (in_dim - k)
                    # that is, the rank is too low 
                    
                    remaining_dims = ssr.repeat((len(early_combos),1))
                    
                    for i in range(len(remaining_dims)):
                        remaining_dims[i, early_combos[i]]=-1 
                    
                    #remaining_dims[torch.arange(len(early_combos)).repeat_interleave(len(early_combos[0])), torch.flatten(early_combos)]=-1 

                    
                    total_ones = tensor_region_image_dimension(remaining_dims, architecture, device=device)
                    
                    good_initial_BHs = total_ones >= in_dim - k
                    
                    good_early_combos = early_combos[good_initial_BHs]

                    temporary_maps = torch.vstack([early_layer_maps, last_layer])
                    temporary_biases = torch.vstack([torch.reshape(early_layer_biases, [-1,1]), torch.reshape(last_biases,[-1,1])])

                    total_combos = torch.hstack([good_early_combos.repeat((len(last_combos),1)), last_combos.repeat_interleave(len(good_early_combos),dim=0)+old_vals])
                    points = torch.linalg.solve(temporary_maps[total_combos], -temporary_biases[total_combos])
                    
                    all_points.append(points.reshape([-1,in_dim]))
                    combos.extend(total_combos)
                
            else: 

                #turn early_layer_maps and last_layer into one stack 
                
                temporary_maps = torch.vstack([early_layer_maps, last_layer])

                temporary_biases = torch.vstack([torch.reshape(early_layer_biases, [-1,1]), torch.reshape(last_biases,[-1,1])])
                
                total_combos = torch.hstack([early_combos.repeat((len(last_combos),1)), last_combos.repeat_interleave(len(early_combos),dim=0)+old_vals])

                points = torch.linalg.solve(temporary_maps[total_combos], -temporary_biases[total_combos])
                
                all_points.append(points.reshape([-1,in_dim]))
                combos.extend(total_combos)
                
                
        
        if len(all_points)>0: 
            all_points=torch.vstack(all_points)
        
        #add in intersections of only the last layer 
        #UNLESS the most recent output is singular. 
        
        
        if len(last_biases)<in_dim or image_dim < in_dim:
            pass
        
        else:
            last_combos = torch.combinations(n_out, r=in_dim)
            
            temp_points = torch.linalg.solve(last_layer[last_combos],-last_biases[last_combos])
                        
            all_points = torch.vstack([all_points,temp_points])
            
            last_combos = list((last_combos+old_vals))

            combos.extend(last_combos)
        
        
        return all_points, combos
        
                
def region_image_dimension(temp_ssr, architecture, depth=None): 
    
    if depth is None: 
        depth = len(architecture)-2
    # all top-dim regions begin at n_0-dimensional
    current_dim = architecture[0]
    
    # need to get sign sequences corresponding to individual layers
    cumulative_widths = [0]+[sum(architecture[1:i]) for i in range(2,len(architecture))]
    
    #loop through layer widths until depth
    for i,layerwidth in enumerate(architecture[1:depth+1]): 
        
        #get sign sequence corresponding with most recent layer 
        layer_neurons = temp_ssr[cumulative_widths[i]:cumulative_widths[i+1]]
        
        # the number of 1's in the sign sequence is the 
        # maximum dimension of the image of the region 
        
        max_dim = sum([s == 1 for s in layer_neurons])
                                  
        # generically the dimension will either stay the same 
        # or be collapsed to the dimension of the image of the map
        
        # eg a 1d subspace of R^2 is sent to all of R under a linear map
        # R^2->R unless it is in the kernel of the map which is a 
        # nongeneric condition
         
        current_dim = min(current_dim,max_dim)
    
    return int(current_dim)


def tensor_region_image_dimension(tensor_ssr, architecture, device): 
    
    # all top-dim regions begin at n_0-dimensional
    current_dim = torch.tensor([architecture[0]],device=device).repeat(len(tensor_ssr))
    
    # need to get sign sequences corresponding to individual layers
    cumulative_widths = [0]+[sum(architecture[1:i]) for i in range(2,len(architecture))]
    
    #find the depth to stop at     
    depth = torch.where(torch.Tensor(cumulative_widths)==len(tensor_ssr[0]))[0][0].numpy()
    
    #loop through layer widths until depth
    for i,layerwidth in enumerate(architecture[1:depth+1]): 
        
        #get sign sequence corresponding with most recent layer 
        layer_neurons = tensor_ssr[:,cumulative_widths[i]:cumulative_widths[i+1]]
        
        # the number of 1's in the sign sequence is the 
        # maximum dimension of the image of the region 
        
        max_dim = torch.sum(layer_neurons==1, axis=1)
                                  
        # generically the dimension will either stay the same 
        # or be collapsed to the dimension of the image of the map
        
        # eg a 1d subspace of R^2 is sent to all of R under a linear map
        # R^2->R unless it is in the kernel of the map which is a 
        # nongeneric condition
        current_dim = torch.min(torch.vstack([current_dim,max_dim]),axis=0).values
    
    return current_dim

def make_sphere(dim,n): 
    
    temppts = torch.normal(0.0,1.0,(n,dim))  
    points0 = 2*temppts/torch.linalg.norm(temppts,dim=1).reshape(n,1)  
    scatter = 0.05*torch.normal(0,1,(n,1)) 
    points0=points0*scatter+points0
    
    points1 = 0.2*torch.normal(0.0,1.0, (n,dim))
    
    points=torch.vstack([points0,points1])
    labels=torch.hstack([torch.zeros(n),torch.ones(n)]).reshape(2*n,1)
    
    return points,labels

def make_torus(n): 
        
    thetas = 2*torch.pi*torch.rand(n)
    phis = 2*torch.pi*torch.rand(n)
    
    xs = 4*torch.cos(thetas)-2*torch.cos(thetas)*torch.cos(phis)
    ys = 4*torch.sin(thetas)-2*torch.sin(thetas)*torch.cos(phis)
    zs = 2*torch.sin(phis)
    
    pts0 = torch.vstack([xs,ys,zs]).T
    
    pts1,_ = make_sphere(2,n)
    
    pts1 = 2*torch.hstack([pts1[0:n],torch.zeros((n,1))])
    
    
    points = torch.vstack([pts0,pts1])
    labels=torch.hstack([torch.zeros(n),torch.ones(n)]).reshape(2*n,1)
    
    return points,labels

def get_sse(all_ssv, length):
    
    locs = torch.where(all_ssv==0)
    
    all_sse = []
    
    for loc0, loc1 in zip(*locs): 
        # only do this if loc1 is < length since only want cofaces for ssv
        # up to the length given 
        
        if loc1 < length: 

            temp=torch.clone(all_ssv[loc0])
            temp[loc1]=1 
            
            all_sse.append(torch.clone(temp)) 

            temp[loc1]=-1
            all_sse.append(torch.clone(temp)) 
        
    all_sse=torch.vstack(all_sse)[:,0:length] 
    all_sse = torch.unique(all_sse,dim=0)
    
    return all_sse


def get_new_combos(facecombos, k): 
    # to be a new combo it has to be a k-combo from two k-1 combos 
    # there's more but this is sufficient to narrow it down a lot
    
    goodcombos=[]
    
    #can this be done without loops 
    
    for i, combo1 in enumerate(facecombos): 
        for combo2 in facecombos[i+1:]:
            
            ncombo = torch.unique(torch.hstack([combo1,combo2]))

            if len(ncombo)==k: 
                goodcombos.append(ncombo)
                
    if len(goodcombos)>0: 
        goodcombos = torch.unique(torch.vstack(goodcombos),dim=0)
        
    return goodcombos

def get_intersections_by_combos(region_maps, old_combo, new_combos, temp_ssr, depth, length): 

    #get region maps associated with BH which intersect to form the edge 

    #obtain the early layer maps as a list  of weights and biases
    early_layer_maps, early_layer_biases=[],[]

    for j in range(depth):
        ll, bb = make_linear(region_maps[j])
        early_layer_maps.extend(ll) 
        early_layer_biases.extend(bb) 

    #restrict to those layer maps corresponding to the desired hyperplanes
    early_layer_maps = torch.vstack(early_layer_maps)[old_combo]
    early_layer_biases = torch.vstack(early_layer_biases)[old_combo]

    #obtain the last layer maps as a list of weights and biases 
    last_layer, last_biases = make_linear(region_maps[-1])

    # put all the maps together. Probably too many loops.
    # get the point corresponding to each of the next hyperplanes intersecting
    # with this edge 


    affmaps = []
    biases = []
    combos = []
    
    for combo in new_combos:
        affmap = torch.vstack([early_layer_maps,
                               last_layer[combo]])


        bias = torch.vstack([early_layer_biases,
                             last_biases[combo].reshape([-1,1])]).reshape([-1])


        affmaps.append(affmap[None,:])
        biases.append(bias)

        combo = torch.hstack([old_combo, combo+length])
        combos.append(combo)
    
    temp_points = torch.linalg.solve(torch.vstack(affmaps),-torch.vstack(biases))
    temp_combos = torch.vstack(combos).long()
        
    return temp_points, temp_combos

def create_append_torch(tensor_of_keys, list_of_items, key_to_add, thing_to_add ): 
    #treats a torch tensor as a dictionary
    
    # get the location where this exists 
    loc = torch.where((tensor_of_keys == key_to_add).all(dim = 1))[0]
    
    if len(loc)==0: 
        tensor_of_keys = torch.vstack([tensor_of_keys, key_to_add])
        list_of_items.append([thing_to_add])
        
    elif len(loc)==1:
        list_of_items[loc[0]].append(thing_to_add)
        
    elif len(loc)>1: 
        raise ValueError("Multiple key locations. Terminating.")
    
    return tensor_of_keys

def get_sse_coboundary_torch(all_ssv, length, verbose=False):
        
    locs = torch.where(all_ssv==0)
    
    # make a placeholder so it's always appendable then delete later. 
    # is there a better way to do this?  
    
    all_sse = torch.tensor([[0]*length]) 
    sse_coboundary_list = [list()]  
    
    for loc0, loc1 in zip(*locs): 
        # only do this if loc1 is < length since only want cofaces for ssv
        # up to the length given 
        
        if loc1 < length: 
            
            temp=torch.clone(all_ssv[loc0])
            temp[loc1]=1 
            
            #create_append_torch(tensor_of_keys, list_of_items, key_to_add, thing_to_add)
            all_sse = create_append_torch(all_sse, sse_coboundary_list, temp[:length], all_ssv[loc0]) 

            temp[loc1]=-1
            
            all_sse = create_append_torch(all_sse, sse_coboundary_list, temp[:length], all_ssv[loc0]) 
            
    
    return all_sse[1:], sse_coboundary_list[1:]

def thicken_model(model_in, threshold_shifts): 
    # currently applies to DeepNeuralNetwork (k,n,n,1) only. 
    # threshold_shifts should be a list of the form [low shift, upper shift] 
    # where the first number is negative and the second number is positive. 
    
    sd = model_in.state_dict()
    new_arch = list(model_in.architecture)
    new_arch[-1] = new_arch[-1]+1
    new_arch = tuple(new_arch)
    new_arch
    
    thickened_model = DeepNeuralNetwork(new_arch)
    
    params = [] 
    for p in model_in.parameters(): 
        params.append(p.data)
    
    params[-2] = params[-2].repeat((2,1))
    params[-1] = params[-1].repeat(2)
    params[-1] += torch.Tensor(threshold_shifts)
    
    sd["linear_2.weight"] = params[-2]
    sd["linear_2.bias"] = params[-1] 
    
    thickened_model.load_state_dict(sd)
    
    return thickened_model

def add_bbox(model_in, bbox):
    # currently applies to DeepNeuralNetwork (k,n,n,l) only. 
    # bbox should be a positive float and the bounding box will be 
    # added at +- bbox in each dimension.

    sd = model_in.state_dict()
    new_arch = list(model_in.architecture)
    
    #add a positive and negative hyperplane in each dimension 
    new_arch[1] = new_arch[1]+2*(model_in.architecture[0])
    new_arch = tuple(new_arch)
    
    boxed_model = DeepNeuralNetwork(new_arch)

    params = [] 
    for p in model_in.parameters(): 
        params.append(p.data)
        
    old_weight = sd["linear_0.weight"]

    extra_weight = torch.zeros((model_in.architecture[0]*2, model_in.architecture[0]))

    old_bias = sd["linear_0.bias"] 

    extra_bias = torch.zeros(model_in.architecture[0]*2)

    for k in range(model_in.architecture[0]): 
        extra_weight[2*k,k] = 1
        extra_weight[2*k+1,k] = 1
        extra_bias[2*k] = -bbox
        extra_bias[2*k+1] = bbox

    extra_weight = extra_weight + torch.randn(extra_weight.shape)*.0001

    new_weight = torch.vstack((extra_weight,old_weight))
    new_bias = torch.hstack((extra_bias,old_bias))


    sd["linear_0.weight"] = new_weight
    sd["linear_0.bias"] = new_bias

    disregarded_weights = torch.randn((model_in.architecture[2],2*model_in.architecture[0]))*.0001

    #print(sd["linear_1.weight"], disregarded_weights)
    sd["linear_1.weight"] = torch.hstack((disregarded_weights,sd["linear_1.weight"]))

    boxed_model.load_state_dict(sd)
    
    return boxed_model

def get_full_complex(model_in, 
                        max_depth=None, 
                        device=None, 
                        mode='solve', 
                        verbose=False, 
                        get_unbounded=False, 
                        thickdb=False, 
                        bbox=False): 
    '''assumes model is feedforward and has appropriate structure.
    Outputs dictionary with vertices' signs and locations of vertices. 
    If get_unbounded is True, then also returns a second dictionary, 
    with unbounded edges' sign sequences, 
    their starting vertex, and their direction.'''
                           
    #do some surgery on the last layer in the case of thickdb. 
    if thickdb: 
        model = thicken_model(model_in, thickdb) 
    else: 
        model = model_in
                            
    if bbox: 
        model = add_bbox(model,bbox)
        
    if device is None: 
        device='cpu'
    
    parameters = list(model.parameters())
    
    if max_depth is None: 
        depth = len(parameters)//2 
    else: 
        depth = max_depth
    
    architecture = [parameters[0].shape[1]] #input dimension 
                   
    for i in range(depth): 
        architecture.append(parameters[2*i].shape[0]) #intermediate dimensions 
        
    architecture = tuple(architecture) 



    in_dim = architecture[0]
    
    signs = torch.tensor(get_signs(in_dim), device=device).float()

    
    #get first layer sign sequences.
    
    if mode == 'solve':
        temp_points, temp_combos = find_intersections(in_dim,parameters[0],
                                                      parameters[1], None, None,    
                                                      architecture, device=device)

       
    else: 
        print("Mode invalid.")
        
    #initialize full list of points, sign sequences, and ss_dict  
    all_points, all_ssv = determine_existing_points(temp_points,temp_combos,model, device=device)

    #trim outer points from bounding box
    if bbox: 
        for k in range(architecture[0]):
            all_points = all_points[all_ssv[:,2*k]!= all_ssv[:,2*k+1]]
            all_ssv = all_ssv[all_ssv[:,2*k]!= all_ssv[:,2*k+1]]
                            
    if verbose: 
        print("First Layer Complete")

        
    tsv=all_ssv.clone() #.cpu().detach().numpy()
    
    all_ss_dict = {tuple(ss.int()): pt for ss, pt in zip(tsv,all_points)}

    # get subsequent layer sign sequences 
    # requires updating points, ssv and ss_dict 

    nlayers=len(architecture)-1
                            
    #loop through layers 
    for i in range(1, nlayers):

        #intead of obtaining regions which are present from previous layer 
        # first obtain edges which are present! 
        
        sse = get_sse(all_ssv, sum(architecture[1:i+1])) 

        #if bb, trim edges under consideration.

        if bbox: 
            for k in range(architecture[0]):
                sse = sse[sse[:,2*k]!= sse[:,2*k+1]]
                
        #initialize placeholder for new points and ssv 
        new_points, new_ssv = [],[]
        
        # for each dimension 1... n0, 
        # loop through cells of that dimension
        # look for intersections with n_0-k bh from
        # new layer. Only look if it is possible the intersection occurs. 
        
        num_sse = len(sse)
        
        if verbose: 
            print("Next Layer Beginning. \n{} edges to evaluate ... ".format(num_sse))
         
        # start with edges before looping through higher dims 
        
        for counter,temp_sse in enumerate(sse):
            
            # obtain the maps on the region induced by the model at each depth
            # note i = layer depth 
            # here the region is just *a* region which contains the edge
            # parameters = list of model parameters
            
            temp_ssr = torch.clone(temp_sse) 
            
            temp_ssr[temp_ssr==0] = -1
            
            #determine whether edge has been collapsed: 
            total_ones = tensor_region_image_dimension(temp_ssr.reshape((1,len(temp_ssr))), architecture, device=device) 
            
            #skip to next edge if this edge is collapsed
            if total_ones == 0: 
                continue 
            
            #get region maps associated with BH which intersect to form the edge 
            region_maps = get_all_maps_on_region(temp_ssr,i,parameters,architecture, device=device)
            
            which_hyperplanes = torch.where(temp_sse==0)[0]
            
            #obtain the early layer maps as a list  of weights and biases
            early_layer_maps, early_layer_biases=[],[]
            
            for j in range(i):
                ll, bb = make_linear(region_maps[j])
                early_layer_maps.extend(ll) 
                early_layer_biases.extend(bb) 
            
            #restrict to those layer maps corresponding to the desired hyperplanes
            early_layer_maps = torch.vstack(early_layer_maps)[which_hyperplanes]
            early_layer_biases = torch.vstack(early_layer_biases)[which_hyperplanes]
            
            #obtain the last layer map as a list of weights and biases 
            last_layer, last_biases = make_linear(region_maps[-1])
            
            # put all the maps together. TODO: Reduce number of loops 
            # get the point corresponding to each of the next hyperplanes intersecting
            # with this edge 
            
            
            affmaps = []
            biases = []
            combos = []
            
            for num, (lastlayer, lastbias) in enumerate(zip(last_layer, last_biases)):
                affmap = torch.vstack([early_layer_maps,
                                       lastlayer])
                                
                bias = torch.vstack([early_layer_biases,
                                     lastbias]).reshape([-1])
                
                
                affmaps.append(affmap[None,:])
                biases.append(bias)
                
                combo = torch.hstack([which_hyperplanes, torch.tensor([num], device=device)+sum(architecture[1:i+1])])
                combos.append(combo)
                
            temp_points = torch.linalg.solve(torch.vstack(affmaps),-torch.vstack(biases))
            temp_combos = torch.vstack(combos).long()
                                                                    
            #if there's at least one point evaluate whether they belong to C(F)
            
            if len(temp_points)>0: 
                
                #get which points have the appropriate sign sequences 
                temp_pts, temp_ssv = determine_existing_points(temp_points,
                                                               temp_combos, 
                                                               model, 
                                                               region_ss=temp_ssr, 
                                                               device=device)

                new_points.extend(temp_pts)
                new_ssv.extend(temp_ssv)
                
            if verbose:
                print("*"*int((counter+1)/num_sse*20+1), 
                      "."*(20-int((counter+1)/num_sse*20)-1),
                      " {percent:.2f}%".format(percent=(counter+1)/num_sse*100),
                      end='\r')
                
            #done with edges 
        if verbose:
            print("")
        
        # now do for higher dimensions
        # the only possible k-faces with a new vertex 
        # are those adjacent to the ones in new_ssv. 
        
        # note: new_points[0] is now the list of new points from edges
        # new_points[1] will be the list of new points from faces, & so on 
        # new_ssv[0] is the list of sign sequences from previous vertices
        # when restricted to first n0 will have 
        # new_ssv[1] is the list of sign sequences from faces, & so on 
        
        if len(new_points)>0: 

            new_points_by_dim = [torch.vstack(new_points)]
            new_ssv_by_dim = [torch.vstack(new_ssv)]
        else: 
            new_points_by_dim = [list()]
            new_ssv_by_dim = [list()]
            
            # get k-faces with a possible new intersection in their interior, 
            # new intersection of n_0-k "new hyperplanes". 
            # for this to occur need n_0-k-1 "new hyperplanes" from a k-1 face 
            # incident to this k-face. 
            
            # previous iteration of loop (or edges) got k-1 faces and n_0-k-1 combos
            # in the form of the ssv values 
            
            # create dictionary with (k-faces) : (k-1)-combos and 
            # (k-faces): 1-combos 
            
            # use sse and ssr as before except sse might not be the ss of an edge, just a k-face

            
        for k in range(2,min(in_dim+1, architecture[i+1]+1)): 
            if verbose:
                print("finding intersections of {} new hyperplanes with {} old hyperplanes".format(k, in_dim-k))
            # here k is the dimension of the face in C(F_i) we are intesecting
            # with n_0-k other hyperplanes. Can't do this if there aren't n_0-k hyperplanes
            # to intersect! 
            
            if len(new_points_by_dim[-1])==0 or (thickdb and i==nlayers-1): #skip this in the case of thickened decision boundary.
                break
            else:
                pass
            
            new_points = []
            new_ssv = []
            
            length = sum(architecture[1:i+1])
            
            # get next dimension up regions which are adjacent to most recently added vertices
            
            sse, sse_coboundary = get_sse_coboundary_torch(new_ssv_by_dim[-1], length)
                        
            
            # loop through eligible k-faces 
            
            for faceloc, kface in enumerate(sse): 
                
                # get ssr. 
                temp_ssr = torch.clone(kface) 
            
                #get a region which is adjacent to the edge; lowest dim region will do. 
                temp_ssr[temp_ssr==0] = -1
                
                #print(temp_ssr, kface)

                #determine whether region has been collapsed.  
                total_ones = tensor_region_image_dimension(
                    temp_ssr.reshape((1,len(temp_ssr))), 
                    architecture, 
                    device=device) 
                
                #skip to next region if this region is 
                # collapsed to lower dim than # hyperplanes intersecting
                
                if total_ones < k: 
                    #print("skipped a point in region {}".format(temp_ssr.detach().numpy()))
                    continue 
                
                #now do the intersections
                
                vcombos = []
                
                # get which combos of hyperplanes intersect on the boundary
                # of the kface 

                # find sse location in sse coboundary 
                
                # these are the previous edges which the vertices were found in
                for v in sse_coboundary[faceloc]: 
                    vcombo = torch.where(v==0)
                    vcombos.append(vcombo[0][vcombo[0]>=length])
                
                #these combos of k-1 new hyperplanes appear on a face of the kface 
                face_combos = torch.unique(torch.vstack(vcombos),dim=0)
                
                #this is the combo of bh which intersect to form the kface
                old_combo = torch.where(kface[:length]==0)[0]
                
                #to be a new combo it has to be a k-combo from two k-1 combos on the edge of kface
                new_combos = get_new_combos(face_combos, k) 
                #print(len(new_combos))
                
                #if there isn't anything to solve for then don't do it
                if len(new_combos)==0: 
                    continue 
                                
                #if there is anything to solve for then solve for it
                region_maps = get_all_maps_on_region(temp_ssr,i,parameters,architecture, device=device)
                
                new_combos_indices = new_combos - length
                
                
                temp_points, temp_combos = get_intersections_by_combos(region_maps, 
                                                                       old_combo, 
                                                                       new_combos_indices, 
                                                                       temp_ssr,
                                                                       i, 
                                                                       length)
                
                # get which ones are in the correct regions 
                if len(temp_points)>0: 
                
                    #get which points have the appropriate sign sequences 
                    temp_pts, temp_ssv = determine_existing_points(temp_points,
                                                                   temp_combos, 
                                                                   model, 
                                                                   region_ss=temp_ssr, 
                                                                   device=device)

                    new_points.extend(temp_pts)
                    new_ssv.extend(temp_ssv)
                    
                
                if verbose:
                    print("*"*int((faceloc+1)/len(sse)*20+1), 
                          "."*(20-int((faceloc+1)/len(sse)*20)-1),
                          " {percent:.2f}%".format(percent=(faceloc+1)/len(sse)*100),
                          end='\r')
           
            
            
            #done with all k-faces, next dim up 
            if len(new_points)>0: 
                new_points_by_dim.append(torch.vstack(new_points))
                new_ssv_by_dim.append(torch.vstack(new_ssv))
            else: 
                break
        
        
        
        #done looping through regions, now collect points 
       
        if verbose: 
            print("\n Layer {} complete.".format(i+1))
        
        
        if len(new_points_by_dim[0])>0: 
            
            new_points_this_layer=torch.vstack(new_points_by_dim)
            
            
            all_points = torch.vstack([all_points,new_points_this_layer])
            
            new_ssv_this_layer = torch.vstack(new_ssv_by_dim)
            all_ssv = torch.vstack([all_ssv,new_ssv_this_layer]) 
    
            new_ss_dict = {tuple(ss):pt for ss,pt in zip(new_ssv_this_layer,new_points_this_layer)}
            
            all_ss_dict = all_ss_dict | new_ss_dict
        
        else:
            pass
        
    
    return all_ss_dict, all_points, all_ssv


def plot_complex(plot_dict, num_comparison, dim, ax=None, colors=None):
    ''' Plots the polyhedral complex with plot_dict in the form {ss:coordinates}'''

    if ax is None: 
        fix,ax =  plt.subplots(figsize=(5,5)) 
        ax.set_xlim((-10,10))
        ax.set_ylim((-10,10))

    
    if colors is None: 
        colors = ['black']*len(plot_dict)
         
    #for each pair of vertices: 
    for v in plot_dict: 
        for w in plot_dict: 

            #determine if they are connected by an edge
            if edge_connected(v[0:num_comparison],w[0:num_comparison], dim=dim): 
                
                hyper_set = set(np.where(np.array(v)==0)[0]).intersection(set(np.where(np.array(w)==0)[0]))
                hyper = max(hyper_set)
                
                #color the edge with the color of the latest hyperplane participating in the edge. 
                color = colors[hyper]
                
                if color=="black" or color=="blue":
                    ax.plot(*np.vstack([plot_dict[v],plot_dict[w]]).T, c=color,alpha=.1,zorder=1)
                elif color=="white": 
                    pass
                else:
                    ax.plot(*np.vstack([plot_dict[v],plot_dict[w]]).T, c=color,alpha=.5,zorder=1)
                    
    return ax

def get_unbounded_edges(model, output, max_depth=None): 
    #currently only works for input dimension 2 
    
    parameters = list(model.parameters())
    
    if max_depth is None: 
        depth = len(parameters)//2 
    else: 
        depth = max_depth

   # print(depth)
    architecture = [parameters[0].shape[1]] #input dimension 
    
    
    for i in range(depth): 
        architecture.append(parameters[2*i].shape[0]) #intermediate dimensions 
        
    architecture = tuple(architecture) 
   

    cumulative_architecture = torch.Tensor([sum(architecture[1:i+1])-1 for i in range(1,len(architecture))])
    
    # actual work 
    
    pd = cx.numpyize_plot_dict(output[0])
    
    #get all edges. 
    sse, sse_coboundary = cx.get_sse_coboundary_torch(output[2], sum(architecture[1:depth+1]))

    #get all unbounded edges 
    
    unbounded_edges = [] 
    unbounded_vertices = [] 
    
    for i, (ss, cob) in enumerate(zip(sse, sse_coboundary)): 
        if len(cob)==1: 
            unbounded_edges.append(ss)
            unbounded_vertices.append(cob[0])

    unbounded_edges = torch.vstack(unbounded_edges) 
    unbounded_vertices = torch.vstack(unbounded_vertices)
    
    # Process: getting the vector of the unbounded edge, up to sign. 

    unbounded_directions = [] 

    for i, ss in enumerate(unbounded_edges): 
        ss_incidence = torch.clone(ss)
        ss_incidence[ss_incidence==0] = 1 

        # first get the layer maps on the incident region.   
        maps = cx.get_all_maps_on_region(ss_incidence, depth-1, parameters, architecture)

        # identify which indices ss==0, and what layer 
        
        index_of_zero = torch.where(ss==0)[0][0]
        
        # get the layer corresponding to this index 
        which_layer = torch.where(cumulative_architecture>=index_of_zero)[0][0]
        #print(which_layer)

        #get the layer map corresponding to this index 
        lmap = maps[which_layer]

      #  print(lmap)
        
        #get edge Linear maps. In 2d there is only one.  
        
       # print(index_of_zero)
        if index_of_zero > cumulative_architecture[0]: 
            all_Hi = lmap[int(index_of_zero - cumulative_architecture[which_layer-1]-1)]
        else: 
            all_Hi = lmap[int(index_of_zero)]
        #print(all_Hi) 

        #Get kerenel of the linear map 
        vector = torch.Tensor([all_Hi[1], -all_Hi[0]])
        vector = vector/torch.linalg.norm(vector)

        #determine whether this vector should be in the same or opposite direction 

        #find the vertex the edge is attached to 
        ssv = unbounded_vertices[i]
        point = pd[tuple(ssv.detach().numpy())]

        #move along that vector from that point 
        eval_point = (point+vector.detach().numpy())

        #evaluate the linear map at that point associated with the vertex's other zero coordinate 
        vertex_zero_indices = torch.where(ssv==0)[0]

        compare_zero_index = vertex_zero_indices[vertex_zero_indices!=index_of_zero] 
        which_layer_compare = torch.where(cumulative_architecture>=compare_zero_index)[0][0]
        lmap = maps[which_layer_compare]
        
        if compare_zero_index > cumulative_architecture[0]: 
            Hi_compare = lmap[int(compare_zero_index - cumulative_architecture[which_layer_compare-1]-1)]
        else: 
            Hi_compare = lmap[int(compare_zero_index)]

        #evaluate eval_point
        sign = sum(Hi_compare.detach().numpy()[0:2] * eval_point) +Hi_compare.detach().numpy()[2]
        
        if sign*ss[compare_zero_index]>0 : 
            vector = vector 
        else: 
            vector = vector*-1 
        
        unbounded_directions.append(vector) 

    unbounded_directions = torch.vstack(unbounded_directions)
    
    return     unbounded_edges, unbounded_vertices, unbounded_directions


def plot_unbounded(pd, ub_e, ub_v, ub_d, ax, colors): 
    '''Currently only works in input dimension 2.'''
    for i in range(len(ub_e)): 
        point = pd[tuple(ub_v[i].detach().numpy())]
        
        color_index = torch.where(ub_e[i]==0)[0][0]
        
        ax.arrow(point[0], point[1], ub_d[i][0].detach().numpy(), ub_d[i][1].detach().numpy(), 
                 color=colors[color_index], zorder=2, head_width = .1)


def plot_fast(vertex_tensor, plot_dictionary, 
              bound = None, colors="red_db", 
              alphas = "default", ax = None, 
              return_vals=False): 
    
    ''' inputs pytorch vertex tensor and plot dictionary directly 
    if using output = cx.get_full_complex, then 
    first argument should be output[2] and plot_dictionary = output[0]. 
    This version doesn't spend so much time deciding what things to plot'''
    
    length = len(vertex_tensor[0])
    
    if bound is None: 
        bound = 10
    
    if ax is None: 
        fig,ax = plt.subplots(1,1, figsize=(5,5))
        
    if colors == "red_db": 
        colors = ['clear']*(length-1) +['red']
        
    if alphas=="default": 
        alphas = [1]*len(vertex_tensor[0])
    
    
    # this gets edges. 
    all_sse, sse_coboundary = get_sse_coboundary_torch(vertex_tensor, length)

    npdict = numpyize_plot_dict(plot_dictionary)
    
    #loop through edges
    for edgeloc, edge in enumerate(all_sse): 
        
        #gets the last hyperplane participating in the intersection. 
        
        color_place = np.max(np.where(edge==0))
        
        c = colors[color_place]
        a = alphas[color_place]
        
        # get the vertices of this edge  
        
        pts = sse_coboundary[edgeloc]
                
        #only plot edge if it is bounded  
        if len(pts)==2:  
            p1 = npdict[tuple(pts[0].detach().numpy())]
            p2 = npdict[tuple(pts[1].detach().numpy())]

            if c == 'clear': 
                pass
            else: 
                ax.plot(*np.stack([p1,p2]).T, c=c, alpha=a)

    ax.set_xlim([-bound,bound])
    ax.set_ylim([-bound,bound])
    
    return ax

### borrowed from stackoverflow ### 
### https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib 

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )
    
def is_critical_point(temppt, pd, model, edges, boundaries): 
    # needs improvement
    # pd=cx.numpyize_plot_dict(output[0])
    
    centerval = model(torch.tensor(np.array([pd[tuple(temppt.detach().numpy())]])))[-1].item()


    locs = torch.where(temppt==0)

    curve_up_directions = []
    curve_down_directions = []
    unbd_directions = [] 
    
    for loc in locs[0]:
        e1 = torch.clone(temppt)
        e1[loc] = 1 

        e2 = torch.clone(temppt) 
        e2[loc] = -1 

        e1_index = torch.where((edges==e1).all(dim=1))[0] 
        e2_index = torch.where((edges==e2).all(dim=1))[0] 

       

        if len(boundaries[e1_index]) == 1 or len(boundaries[e2_index])==1: 
            unbd_directions.append(loc)
            continue
            
        # check whether the network will be constant on the edge using sign sequences
        # if so, v is necessarily a critical point. 
        # these will be classified differently
        
        e1collapsed = torch.clone(e1) 
        
        e1collapsed[e1collapsed==0] = -1 
        
        e2collapsed = torch.clone(e1) 
        
        e2collapsed[e2collapsed==0] = -1 
        
        
        #if the edge has been collapsed: 
        if region_image_dimension(e1collapsed, model.architecture, depth=len(model.architecture)-2) == 0: 
            return True, -1
        elif region_image_dimension(e2collapsed, model.architecture, depth=len(model.architecture)-2) == 0:
            return True, -1
        
         # the location of the other point is the other point in the edge 
        
        p1_loc = 1- torch.where((torch.vstack(boundaries[e1_index] )== temppt).all(dim=1))[0][0]
        p1 = torch.vstack(boundaries[e1_index])[p1_loc].detach().numpy()

        p2_loc = 1- torch.where((torch.vstack(boundaries[e2_index] )== temppt).all(dim=1))[0][0]
        p2 = torch.vstack(boundaries[e2_index])[p2_loc].detach().numpy()

        ptval = model(torch.tensor(np.array([pd[tuple(p1)],pd[tuple(p2)]])))[-1]
        
        if (ptval>centerval).all(): 
            curve_up_directions.append(loc) 
        elif (ptval<centerval).all(): 
            curve_down_directions.append(loc)
        else:
            pass
        

    if len(curve_up_directions)+len(curve_down_directions)==sum(temppt==0): 
        index = len(curve_down_directions)

        return True, index
    else:
        return False, None