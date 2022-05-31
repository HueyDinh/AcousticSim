from multiprocessing import connection
from multiprocessing.sharedctypes import Value
import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag, eig
from dataclasses import dataclass
from abc import ABC, abstractmethod
import cv2

# TypeAliases
CoordList = tuple[list[float], ...]
FloatMatrix = npt.NDArray[np.float64]
Int8GreyScaleImg = npt.NDArray[np.int8]

# Define Input Interface
@dataclass
class Material:

    front_board_k: float
    back_board_k: float
    front_board_diag_spring_ratio: float
    back_board_diag_spring_ratio: float
    post_k: float
    bridge_k: float

    front_board_m: float
    back_board_m: float
    post_m: float
    bridge_m: float

    alpha: float
    beta: float


@dataclass
class Geometry:

    board_profile: Int8GreyScaleImg
    wall_profile: Int8GreyScaleImg
    holes_profile: Int8GreyScaleImg

    bridge_location: Int8GreyScaleImg
    post_location: Int8GreyScaleImg
    
    wall_height: float
    scale_factor: float


@dataclass
class StringTuning:

    string_tension: float
    string_mass_per_length: float
    note_location: float
    input_type: int
    input_fractional_location: float
    input_amplitude: float
    num_node: int


@dataclass
class Simulation:

    simulation_period: float
    sampling_rate: float = 44100


# Define Abstract Concepts
class Vibration(ABC):

    M_matrix: FloatMatrix
    K_matrix: FloatMatrix

    @abstractmethod
    def assemble_K(self):
        pass
            
    @abstractmethod
    def assemble_M(self):
        pass


class ImageBasedData(ABC):

    @abstractmethod
    def parse_pixel_array(self):
        pass


# Define Instrument
class Acoustic:

    # Auxillary Classes
    class BackBoard(Vibration, ImageBasedData):

        wall_elements_pixel: Int8GreyScaleImg
        wall_elements_tuple: CoordList

        interior_elements_pixel: Int8GreyScaleImg
        interior_elements_tuple: CoordList

        k: float
        m: float
        diag_spring_ratio: float

        def __init__(self, m: float, k: float,
                     diag_spring_ratio: float,
                     wall_pixel_data: Int8GreyScaleImg,
                     interior_pixel_data: Int8GreyScaleImg):

            self.wall_elements_pixel = wall_pixel_data
            self.interior_elements_pixel = interior_pixel_data

            self.m = m
            self.k = k
            self.diag_spring_ratio = diag_spring_ratio

            self.parse_pixel_array()
            self.assemble_M()
            self.assemble_K()

            return

        def parse_pixel_array(self) -> None:

            self.wall_elements_tuple = np.argwhere(self.wall_elements_pixel == 0).tolist()
            self.interior_elements_tuple = np.argwhere(self.interior_elements_pixel == 0).tolist()

            return
        
        def assemble_M(self) -> None:

            dimension: int = len(self.interior_elements_tuple)
            self.M_matrix = np.diag(np.full(dimension, self.m))

            return
        
        def assemble_K(self) -> None:

            dimension: int = len(self.interior_elements_tuple)
            global_K: FloatMatrix = np.zeros((dimension, dimension))

            for index, (x_coord, y_coord) in enumerate(self.interior_elements_tuple):

                adj_list: CoordList = (
                    [x_coord, y_coord + 1],
                    [x_coord, y_coord - 1],
                    [x_coord + 1, y_coord],
                    [x_coord - 1, y_coord]
                )

                diag_adj_list: CoordList = (
                    [x_coord + 1, y_coord + 1],
                    [x_coord - 1, y_coord + 1],
                    [x_coord + 1, y_coord - 1],
                    [x_coord - 1, y_coord - 1]
                )

                for adj_coord in adj_list:

                    if adj_coord in self.interior_elements_tuple:

                        adj_index = self.interior_elements_tuple.index(adj_coord)
                        global_K[index,index] += self.k
                        global_K[index,adj_index] -= self.k/2
                        global_K[adj_index,index] -= self.k/2
                    
                    if adj_coord in self.wall_elements_tuple:

                        global_K[index,index] += self.k
                
                for diag_adj_coord in diag_adj_list:

                    if diag_adj_coord in self.interior_elements_tuple:

                        diag_adj_index = self.interior_elements_tuple.index(diag_adj_coord)
                        global_K[index,index] += self.k*self.diag_spring_ratio
                        global_K[index,diag_adj_index] -= self.k/2 * self.diag_spring_ratio
                        global_K[diag_adj_index,index] -= self.k/2 * self.diag_spring_ratio
                    
                    if diag_adj_coord in self.wall_elements_tuple:

                        global_K[index,index] += self.k * self.diag_spring_ratio
            
            self.K_matrix = global_K
            
            return


    class FrontBoard(Vibration, ImageBasedData):

        wall_elements_pixel: Int8GreyScaleImg
        wall_elements_tuple: CoordList

        interior_elements_pixel: Int8GreyScaleImg
        interior_elements_tuple: CoordList

        hole_elements_pixel: Int8GreyScaleImg
        hole_elements_tuple: CoordList

        k: float
        m: float
        diag_spring_ratio: float

        def __init__(self, m: float, k: float,
                     diag_spring_ratio: float,
                     wall_pixel_data: Int8GreyScaleImg,
                     interior_pixel_data: Int8GreyScaleImg,
                     hole_pixel_data: Int8GreyScaleImg):

            self.wall_elements_pixel = wall_pixel_data
            self.interior_elements_pixel = interior_pixel_data
            self.hole_elements_pixel = hole_pixel_data

            self.m = m
            self.k = k
            self.diag_spring_ratio = diag_spring_ratio

            self.parse_pixel_array()
            self.assemble_M()
            self.assemble_K()

            return

        def parse_pixel_array(self) -> None:

            self.wall_elements_tuple = np.argwhere(self.wall_elements_pixel == 0).tolist()
            self.interior_elements_tuple = np.argwhere((self.interior_elements_pixel == 0) & (self.hole_elements_pixel != 0)).tolist()
            self.hole_elements_tuple = np.argwhere(self.hole_elements_pixel == 0).tolist()

            return
        
        def assemble_M(self) -> None:

            dimension: int = len(self.interior_elements_tuple)
            self.M_matrix = np.diag(np.full(dimension, self.m))

            return
        
        def assemble_K(self) -> None:

            dimension: int = len(self.interior_elements_tuple)
            global_K: FloatMatrix = np.zeros((dimension, dimension))

            for index, (x_coord, y_coord) in enumerate(self.interior_elements_tuple):

                adj_list: CoordList = (
                    [x_coord, y_coord + 1],
                    [x_coord, y_coord - 1],
                    [x_coord + 1, y_coord],
                    [x_coord - 1, y_coord]
                )

                diag_adj_list: CoordList = (
                    [x_coord + 1, y_coord + 1],
                    [x_coord - 1, y_coord + 1],
                    [x_coord + 1, y_coord - 1],
                    [x_coord - 1, y_coord - 1]
                )

                for adj_coord in adj_list:

                    if adj_coord in self.interior_elements_tuple:

                        adj_index = self.interior_elements_tuple.index(adj_coord)
                        global_K[index,index] += self.k
                        global_K[index,adj_index] -= self.k/2
                        global_K[adj_index,index] -= self.k/2
                    
                    if adj_coord in self.wall_elements_tuple:

                        global_K[index,index] += self.k
                
                for diag_adj_coord in diag_adj_list:

                    if diag_adj_coord in self.interior_elements_tuple:

                        diag_adj_index = self.interior_elements_tuple.index(diag_adj_coord)
                        global_K[index,index] += self.k*self.diag_spring_ratio
                        global_K[index,diag_adj_index] -= self.k/2 * self.diag_spring_ratio
                        global_K[diag_adj_index,index] -= self.k/2 * self.diag_spring_ratio
                    
                    if diag_adj_coord in self.wall_elements_tuple:

                        global_K[index,index] += self.k * self.diag_spring_ratio
            
            self.K_matrix = global_K
            
            return


    class Post(Vibration, ImageBasedData):
        
        M_bulk: float
        K_bulk: float

        location_pixel: Int8GreyScaleImg
        location_tuple: CoordList

        def __init__(self, K_bulk: float, M_bulk: float, post_location_image: Int8GreyScaleImg):
            
            self.K_bulk = K_bulk
            self.M_bulk = M_bulk
            self.location_pixel = post_location_image

            self.parse_pixel_array()
            self.assemble_M()
            self.assemble_K()

            return
        
        def parse_pixel_array(self) -> None:

            self.location_tuple = np.argwhere(self.location_pixel == 0).tolist()

            return
        
        def assemble_M(self):

            self.M_matrix = np.array(self.M_bulk).reshape((1,1))

            return
        
        def assemble_K(self):

            self.K_matrix = np.array(self.K_bulk).reshape((1,1))

            return


    class Bridge(Vibration, ImageBasedData):

        M_bulk: float
        K_bulk: float

        location_pixel: Int8GreyScaleImg
        location_tuple: CoordList

        def __init__(self, K_bulk: float, M_bulk: float, bridge_location_image: Int8GreyScaleImg):
            
            self.K_bulk = K_bulk
            self.M_bulk = M_bulk
            self.location_pixel = bridge_location_image

            self.parse_pixel_array()
            self.assemble_M()
            self.assemble_K()

            return
        
        def parse_pixel_array(self) -> None:

            self.location_tuple = np.argwhere(self.location_pixel == 0).tolist()

            return
        
        def assemble_M(self):

            self.M_matrix = np.array(self.M_bulk).reshape((1,1))

            return
        
        def assemble_K(self):

            self.K_matrix = np.array(self.K_bulk).reshape((1,1))

            return


    class String(Vibration):

        string_tension: float
        string_mass_per_length: float
        note_location: float
        input_type: int
        input_fractional_location: float
        input_amplitude: float
        num_node: int

        el_len: float
        eq_k: float


        def __init__(
            self, string_tension: float,
            string_mass_per_length: float,
            note_location: float,
            input_type: int,
            input_fractional_location: float,
            input_amplitude: float,
            num_node: int):
            
            self.string_tension = string_tension
            self.string_mass_per_length = string_mass_per_length
            self.note_location = note_location
            self.input_type = input_type
            self.input_fractional_location = input_fractional_location
            self.input_amplitude = input_amplitude
            self.num_node = num_node

            self.el_len = self.note_location/(self.num_node + 1)
            self.eq_k = self.string_tension/self.el_len
            self.eq_m = self.string_mass_per_length*self.el_len
            self.input_node_index = round(self.num_node * self.input_fractional_location)

            self.assemble_M()
            self.assemble_K()

            return
        
        def assemble_M(self):

            self.M_matrix = np.diag(np.full(self.num_node, self.string_mass_per_length))

            return
        
        def assemble_K(self):

            self.K_matrix = (np.diag(np.full(self.num_node, self.eq_k*2)) 
                           - np.diag(np.full(self.num_node - 1, self.eq_k),1) 
                           - np.diag(np.full(self.num_node - 1, self.eq_k),-1))
            
            self.K_matrix[0,0] -= self.eq_k

            return

    # Object States
    front_board: FrontBoard
    back_board: BackBoard
    bridge: Bridge
    post: Post
    string: String
    simulation: Simulation
    alpha: float
    beta: float

    global_size: int
    front_board_index_offset: int
    back_board_index_offset: int
    bridge_index_offset: int
    post_index_offset: int
    string_index_offset: int

    global_M: FloatMatrix
    global_K: FloatMatrix

    mode_shapes: FloatMatrix
    nat_freq: FloatMatrix

    time_series_response: FloatMatrix

    # Input Type Flags
    PLUCK = 0
    TAP = 1
    BOW = 2


    def __init__(self, design: Geometry, material: Material, tuning: StringTuning, simulation: Simulation):
        
        self.front_board = self.FrontBoard(
            m = material.front_board_m,
            k = material.front_board_k,
            diag_spring_ratio = material.front_board_diag_spring_ratio,
            wall_pixel_data = design.wall_profile,
            interior_pixel_data = design.board_profile,
            hole_pixel_data = design.holes_profile
        )

        self.back_board = self.BackBoard(
            m = material.back_board_m,
            k = material.back_board_k,
            diag_spring_ratio = material.back_board_diag_spring_ratio,
            wall_pixel_data = design.wall_profile,
            interior_pixel_data = design.board_profile
        )

        self.bridge = self.Bridge(
            K_bulk = material.bridge_k,
            M_bulk = material.bridge_m,
            bridge_location_image = design.bridge_location
        )

        self.post = self.Post(
            K_bulk = material.post_k,
            M_bulk = material.post_m,
            post_location_image = design.post_location
        )

        self.string = self.String(
            string_tension = tuning.string_tension,
            string_mass_per_length = tuning.string_mass_per_length,
            note_location = tuning.note_location,
            input_type = tuning.input_type,
            input_fractional_location = tuning.input_fractional_location,
            input_amplitude = tuning.input_amplitude,
            num_node = tuning.num_node
        )

        self.simulation = simulation

        self.assemble_instrument()
        self.analyze_vibration()

        return

    def assemble_instrument(self):

        def global_connection_matrix(global_dimension: int, board_global_coordinate_offset: int, connected_element_global_coordinate: int,
        board_element_tuple: CoordList, connected_element_locations: CoordList, spring_constant: float):

            connection_matrix: FloatMatrix = np.zeros((global_dimension, global_dimension))

            for coordinate in connected_element_locations:
                if coordinate not in board_element_tuple:
                    raise ValueError("Component Not Connected to Board (Pixel NOT OVERLAPPING")
                else:
                    coord_global_index: int = board_global_coordinate_offset + board_element_tuple.index(coordinate)

                    connection_matrix[coord_global_index, coord_global_index] += spring_constant
                    connection_matrix[connected_element_global_coordinate, connected_element_global_coordinate] += spring_constant
                    connection_matrix[connected_element_global_coordinate, coord_global_index] -= spring_constant
                    connection_matrix[coord_global_index, connected_element_global_coordinate] -= spring_constant

            return connection_matrix

        front_board_M = self.front_board.M_matrix
        back_board_M = self.back_board.M_matrix
        bridge_M = self.bridge.M_matrix
        post_M = self.post.M_matrix
        string_M = self.string.M_matrix

        front_board_K = self.front_board.K_matrix
        back_board_K = self.back_board.K_matrix
        bridge_K = self.bridge.K_matrix
        post_K = self.post.K_matrix
        string_K = self.string.K_matrix

        front_board_dim = len(self.front_board.interior_elements_tuple)
        back_board_dim = len(self.back_board.interior_elements_tuple)
        bridge_dim = 1
        post_dim = 1
        string_dim = self.string.num_node

        self.global_size = front_board_dim + back_board_dim + bridge_dim + post_dim + string_dim
        self.front_board_index_offset = 0
        self.back_board_index_offset = self.front_board_index_offset + front_board_dim
        self.bridge_index_offset = self.back_board_index_offset + back_board_dim
        self.post_index_offset = self.bridge_index_offset + bridge_dim
        self.string_index_offset = self.post_index_offset + post_dim

        bridge_board_spring_constant = (self.front_board.k * self.bridge.K_bulk) / (self.front_board.k + self.bridge.K_bulk) / len(self.bridge.location_tuple)
        post_board_spring_constant = (self.front_board.k * self.post.K_bulk) / (self.front_board.k + self.post.K_bulk) / len(self.post.location_tuple)
        string_bridge_spring_constant = (self.string.eq_k * self.bridge.K_bulk) / (self.string.eq_k + self.bridge.K_bulk)

        bridge_front_conn = global_connection_matrix(
            self.global_size, self.front_board_index_offset, self.bridge_index_offset,
            self.front_board.interior_elements_tuple,
            self.bridge.location_tuple,
            bridge_board_spring_constant
        )
                                                    
        bridge_back_conn = global_connection_matrix(
            self.global_size, self.back_board_index_offset, self.bridge_index_offset,
            self.back_board.interior_elements_tuple,
            self.bridge.location_tuple,
            bridge_board_spring_constant
        )

        post_front_conn = global_connection_matrix(
            self.global_size, self.front_board_index_offset, self.post_index_offset,
            self.front_board.interior_elements_tuple,
            self.post.location_tuple,
            post_board_spring_constant
        )

        post_back_conn = global_connection_matrix(
            self.global_size, self.back_board_index_offset, self.post_index_offset,
            self.back_board.interior_elements_tuple,
            self.post.location_tuple,
            post_board_spring_constant
        )

        string_bridge_conn = np.zeros((self.global_size, self.global_size))
        string_bridge_conn[self.string_index_offset, self.string_index_offset] += string_bridge_spring_constant
        string_bridge_conn[self.bridge_index_offset, self.bridge_index_offset] += string_bridge_spring_constant
        string_bridge_conn[self.string_index_offset, self.bridge_index_offset] -= string_bridge_spring_constant
        string_bridge_conn[self.bridge_index_offset, self.string_index_offset] -= string_bridge_spring_constant

        self.global_M = block_diag(front_board_M, back_board_M, bridge_M, post_M, string_M)
        self.global_K = block_diag(front_board_K, back_board_K, bridge_K, post_K, string_K) + bridge_front_conn + bridge_back_conn + post_front_conn + post_back_conn + string_bridge_conn

        return
    
    def analyze_vibration(self):

        self.nat_freq, self.mode_shapes = eig(self.global_K, self.global_M)
        self.nat_freq = np.sqrt(self.nat_freq)

    def diag_state_response(self):

        time_step = 1/self.simulation.sampling_rate
        time_series = np.arange(0, self.simulation.simulation_period, time_step)
        
        if self.string.input_type == Acoustic.BOW:
            pass
        
        elif self.string.input_type == Acoustic.TAP:
            pass

        elif self.string.input_type == Acoustic.PLUCK:
        
            initial_condition_vector =  
            diag_initial_condition_vector = 

            input_matrix = 
            diag_input_matrix =

        return
    
    def element_response(self):

        return
    
    def superimpose_sound_sources(self):
        #TODO
        return

    def export_sound(self, export_path:str, input: Simulation):
        #TODO
        return

if __name__ == "__main__":

    board_profile = cv2.imread("board.png", cv2.IMREAD_GRAYSCALE)
    wall_profile = cv2.imread("wall.png", cv2.IMREAD_GRAYSCALE)
    hole_profile = cv2.imread("hole.png", cv2.IMREAD_GRAYSCALE)
    bridge_profile = cv2.imread("bridge.png", cv2.IMREAD_GRAYSCALE)
    post_profile = cv2.imread("post.png", cv2.IMREAD_GRAYSCALE)

    test_material = Material(
        front_board_k = 1,
        back_board_k = 1,
        front_board_diag_spring_ratio = 0,
        back_board_diag_spring_ratio = 0,
        post_k = 10,
        bridge_k = 5,

        front_board_m = 2,
        back_board_m = 1,
        post_m = 2,
        bridge_m = 1,

        alpha = 0.001,
        beta = 0.002
    )

    test_design = Geometry(
        board_profile = board_profile,
        wall_profile = wall_profile,
        holes_profile = hole_profile,
        bridge_location = bridge_profile,
        post_location = post_profile,
        wall_height = 1,
        scale_factor = 1
    )

    test_tuning = StringTuning(
        string_tension = 71.7846,
        string_mass_per_length = 1.140e-3,
        note_location = 0.635,
        input_type = Acoustic.BOW,
        input_fractional_location = 0.5,
        input_amplitude = 1,
        num_node = 100
    )

    test_input = Simulation(
    simulation_period = 5,
    sampling_rate = 44100
    )

    test_object = Acoustic(test_design, test_material, test_tuning, test_input)
    print()