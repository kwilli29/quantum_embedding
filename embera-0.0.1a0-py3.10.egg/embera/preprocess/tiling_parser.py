import embera

__all__ = ['DWaveNetworkXTiling']

class DWaveNetworkXTiling:
    """ Generate tiling from architecture graph construction. According to
        the architecture family, create a grid of Tile objects.
    """
    def __init__(self, Tg):
        
        # Graph elements
        self.graph = Tg.graph
        self.qubits = list(Tg.nodes)
        self.couplers = list(Tg.edges)
        # Graph dimensions
        m = self.graph["rows"]
        n = self.graph["columns"]
        # Graph type
        family = self.graph['family']
        if family=='chimera':
            self.shape = (m,n)
        elif family=='pegasus':
            self.shape = (3,m,n)
        elif family=='zephyr':
            self.shape = (m,n)
        else:
            raise ValueError("Invalid family. {'chimera', 'pegasus', 'zephyr'}")
        # Graph cooordinates
        dim = len(self.shape)
        labels = self.graph['labels']
        converter = embera.dwave_coordinates.from_graph_dict(self.graph)
        if labels == 'int':
            self.to_nice = converter.linear_to_nice
            self.from_nice = converter.nice_to_linear
        elif labels == 'coordinate':
            self.to_nice = converter.coordinate_to_nice
            self.from_nice = converter.nice_to_coordinate
        elif labels =='nice':
            self.to_nice = lambda n: n
            self.from_nice = lambda n: n
        # Add Tile objects
        self.tiles = {}
        for q in self.qubits:
            tile = self.get_tile(q)
            if tile in self.tiles:
                self.tiles[tile].qubits.append(q)
            else:
                self.tiles[tile] = Tile(tile, self.shape, [q])

    def __iter__(self):
        return self.tiles

    def __getitem__(self, key):
        return self.tiles[key]

    def __delitem__(self, key):
        del self.tiles[key]

    def items(self):
        return self.tiles.items()

    def get_tile(self, x):
        ## KW ##
        # convert Zephyr coords to custom tile coordinate system
        if self.graph['family'] == 'zephyr':
            u,w,j,k,z = x
            if u == 0:
                x_coords = w*0.5
                y_coords = 2*z if k==0 else (2*z)+1
                
            elif u == 1:
                x_coords = z if k==0 else ((2*z)+1)*0.5
                y_coords = w
            else:
                raise ValueError("u must be 0 or 1")
            
            return(x_coords,y_coords)
        ########   
        else:
            t,i,j,u,k = self.to_nice(x) 
            return (t,i,j)[-len(self.shape):] # return i,j

    def set_tile(self, x, tile):
        _,_,_,u,k = self.to_nice(x)
        return self.from_nice((0,)*(3-len(tile)) + tile + (u,k))

    def get_shore(self, x):
        _,_,_,u,_ = self.to_nice(x)
        return u

    def set_shore(self, x, u):
        t,i,j,_,k = self.to_nice(x)
        return self.from_nice((t,i,j,u,k))

    def get_k(self, x):
        _,_,_,_,k = self.to_nice(x)
        return k

    def set_k(self, x, k):
        t,i,j,u,_ = self.to_nice(x)
        return self.from_nice((t,i,j,u,k))

    def get_qubits(self, tile, shore=None, k=None):
        shores = (0,1) if shore is None else (shore,)
        indices = range(self.graph['tile']) if k is None else (k,)
        nice_tile = (0,)+tile if len(tile)==2 else tile
        for u in shores:
            for k in indices:
                n = nice_tile + (u,) + (k,)
                yield self.from_nice(n)

class Tile:
    """ Tile Class """
    def __init__(self, index, shape, qubits):
        self.index = index
        self.qubits = qubits
        ## KW ##
        self.nodes = set()
        ########

    @property
    def supply(self):
        return self.qubits

    ## KW ##
    # taken from old github code from Apr 20, 2020
    # to see newer version of neighbors look to recent Emebera github updates
    
    def neighbors(self, i, j, m, n):
        """ Calculate indices and names of negihbouring tiles to use recurrently
            during migration and routing.
            The vicinity parameter is later used to prune out the neighbors of
            interest.
            Uses cardinal notation north, south, west, east
        """

        north = (i-1,j)   if (i > 0)      else   None # (i-1,j)
        south = (i+1,j)   if (i < m-1)    else   None # (i+1,j)
        west =  (i,j-1)   if (j > 0)      else   None # (i,j-1)
        east =  (i,j+1)   if (j < n-1)    else   None # (i,j+1)

        nw = (i-1,j-1)  if (i > 0    and j > 0)    else None # (i-1,j-1)
        ne = (i-1,j+1)  if (i > 0    and j < n-1)  else None # (i-1,j+1)
        se = (i+1,j+1)  if (i < m-1  and j < n-1)  else None # (i+1,j+1)
        sw = (i+1,j-1)  if (i < m-1  and j > 0)    else None # (i+1,j-1)

        neighbors = [north, south, west, east, nw, ne, se, sw]

        return neighbors
    ########

    def links(self, tile, edges):
        for q in self.qubits:
            for p in tile.qubits:
                if (q,p) in edges:
                    yield (q,p)

    def is_connected(self, tile, edge_list):
        return any(self.links(tile,edge_list))

    def __repr__(self):
        return str(self.qubits)

    def __str__(self):
        return str(self.index)
