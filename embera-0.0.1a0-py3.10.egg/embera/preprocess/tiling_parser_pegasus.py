## KW ##
# Adapted tiling_parser.py to be able to process the neighbor and 3 coordinate tile system of Pegasus
########

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
        elif family =='zephyr':
            self.shape(m,n)
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
        t,i,j,u,k = self.to_nice(x)
        return (t,i,j)[-len(self.shape):]

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
    '''
    def Zget_tile_neighbors(self, tile):
        #### KW ####
        #neighbors = set()
        neighbors = []
        ####
        
        for i, d in enumerate(tile):
            ## KW ## 
            if(isinstance(d, int)):
                # neg = tile[0:i] + (d-1,) + tile[i+1:]
                neg = tile[0:i] + [(d-1,)] + tile[i+1:]
                
                #neighbors.add(neg)
                neighbors.append(neg)
                #pos = tile[0:i] + (d+1,) + tile[i+1:]
                pos = tile[0:i] + [(d+1,)] + tile[i+1:]
                
                neighbors.append(pos)
                #neighbors.add(pos)
            ########

        print(neighbors)
        return [tile for tile in neighbors if tile in self.tiles]
    
    def Yget_tile_neighbors(self,tile):
            i,j = tile
            neighbors = [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]
            return [tile if tile in self.tiles else (-1,-1) for tile in neighbors ]
    '''
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
    # pasted from old github code - Apr 20, 2020
    # changed to  fit a valid neighbor system of pegasus

    def neighbors(self, d, i, j, m, n):
        """ Calculate indices and names of negihbouring tiles to use recurrently
            during migration and routing.
            The vicinity parameter is later used to prune out the neighbors of
            interest.
            Uses cardinal notation north, south, west, east
        """
       
        if (i > 0 and j < n-1 and d==2) or (i>=0 and j<=n-1 and d!=2):
            top = (0, i-1, j+1) if d==2 else (d+1, i, j)
        else: top = None
            
        if (i < m-1 and j > -1 and d==0) or (i<=m-1 and j >= -1 and d!=0):
            bottom = (2,i+1,j-1) if d==0 else (d-1, i, j)
        else: bottom = None

        if (j >-1 and i<=m-1 and i >= 0 and d==0) or (i > 0 and j <= n-1 and j >= -1 and d!=0):
            top_right = (2,i,j-1) if d==0 else (d-1,i-1,j)
        else: top_right = None
    
        if (i < m-1 and j <= n-1 and j >= -1 and d==0) or (j < n-1 and i<=m-1 and i >= 0 and d != 0):
            top_left = (2,i+1,j) if d==0 else (d-1,i,j+1)
        else: top_left = None

        if (j < n-1 and i<=m-1 and i >= 0 and d==2) or (i < m-1 and j <= n-1 and j >= -1 and d!=2):
            bottom_right = (0,i,j+1) if d==2 else (d+1,i+1,j)
        else: bottom_right = None

        if (i > 0 and j <= n-1 and j >= -1 and d==2) or (j > -1 and i<=m-1 and i >= 0 and d!=2):
            bottom_left = (0,i-1,j) if d==2 else (d+1,i,j-1)
        else: bottom_left = None

        neighbors = [top, bottom, top_left, top_right, bottom_right, bottom_left]
        
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
