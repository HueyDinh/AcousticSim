from typing import Callable
import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag, eigh
from dataclasses import dataclass
from abc import ABC, abstractmethod
import cv2
import itertools

# TypeAliases
CoordList = tuple[list[float], ...]
FloatMatrix = npt.NDArray[np.float64]
Int8GreyScaleImg = npt.NDArray[np.int8]

# Define Input Interface
@dataclass
class Material:

    front_board_k: float
    back_board_k: float
    front_board_k_diag: float
    back_board_k_diag: float
    post_k: float
    bridge_k: float

    front_board_density: float
    back_board_density: float
    post_m: float
    bridge_m: float

    alpha: float
    beta: float


@dataclass
class Geometry:

    board_profile: Int8GreyScaleImg
    wall_profile: Int8GreyScaleImg
    holes_profile: Int8GreyScaleImg
    board_thickness: float
    chamber_height: float

    bridge_location: Int8GreyScaleImg
    post_location: Int8GreyScaleImg

    meter_per_pixel: float
    


@dataclass
class StringTuning:

    string_tension: float
    string_mass_per_length: float
    eff_string_length: float
    num_node: int


@dataclass
class NoteSimulation:

    input_type: int
    input_fractional_location: float
    input_amplitude: float

    simulation_period: float
    sampling_rate: int = 44100
    acoustic_gain: float = 1e6


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

# Define Supporting Classes
class ConnectedSystem(Vibration):


    class BackBoard(Vibration, ImageBasedData):

        wall_elements_pixel: Int8GreyScaleImg
        wall_elements_tuple: CoordList

        interior_elements_pixel: Int8GreyScaleImg
        interior_elements_tuple: CoordList

        k: float
        k_diag: float
        m: float

        def __init__(self, m: float, k: float,
                     k_diag: float,
                     wall_pixel_data: Int8GreyScaleImg,
                         interior_pixel_data: Int8GreyScaleImg):

            self.wall_elements_pixel = wall_pixel_data
            self.interior_elements_pixel = interior_pixel_data

            self.m = m
            self.k = k
            self.k_diag = k_diag

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
                        global_K[index,index] += self.k_diag
                        global_K[index,diag_adj_index] -= self.k_diag/2
                        global_K[diag_adj_index,index] -= self.k_diag/2

                    if diag_adj_coord in self.wall_elements_tuple:

                        global_K[index,index] += self.k_diag

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
        k_diag: float
        m: float
    
        def __init__(self, m: float, k: float,
                     k_diag: float,
                     wall_pixel_data: Int8GreyScaleImg,
                     interior_pixel_data: Int8GreyScaleImg,
                     hole_pixel_data: Int8GreyScaleImg):
    
            self.wall_elements_pixel = wall_pixel_data
            self.interior_elements_pixel = interior_pixel_data
            self.hole_elements_pixel = hole_pixel_data
    
            self.m = m
            self.k = k
            self.k_diag = k_diag
    
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
                        global_K[index,index] += self.k_diag
                        global_K[index,diag_adj_index] -= self.k_diag/2
                        global_K[diag_adj_index,index] -= self.k_diag/2
                    
                    if diag_adj_coord in self.wall_elements_tuple:
                    
                        global_K[index,index] += self.k_diag
            
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

        num_node: int
        el_len: float
        eq_k: float
        eff_string_length: float

        def __init__(
            self, string_tension: float,
            string_mass_per_length: float,
            eff_string_length: float,
            num_node: int):

            self.num_node = num_node
            self.eff_string_length = eff_string_length

            self.el_len = self.eff_string_length/(self.num_node + 1)
            self.eq_k = string_tension/self.el_len
            self.eq_m = string_mass_per_length*self.el_len

            self.assemble_M()
            self.assemble_K()

            return

        def assemble_M(self):

            self.M_matrix = np.diag(np.full(self.num_node, self.eq_m))

            return

        def assemble_K(self):

            self.K_matrix = (np.diag(np.full(self.num_node, self.eq_k*2)) 
                           - np.diag(np.full(self.num_node - 1, self.eq_k),1) 
                           - np.diag(np.full(self.num_node - 1, self.eq_k),-1))

            self.K_matrix[0,0] -= self.eq_k

            return

    front_board: FrontBoard
    back_board: BackBoard
    bridge: Bridge
    post: Post
    string: String

    global_size: int
    front_board_index_offset: int
    back_board_index_offset: int
    bridge_index_offset: int
    post_index_offset: int
    string_index_offset: int

    global_M: FloatMatrix
    global_K: FloatMatrix
    global_alpha: float
    global_beta: float

    def __init__(self, front_board: FrontBoard, back_board: BackBoard, bridge: Bridge, post: Post, string: String, global_alpha: float, global_beta: float) -> None:

        self.front_board = front_board
        self.back_board = back_board
        self.bridge = bridge
        self.post = post
        self.string = string
        self.global_alpha = global_alpha
        self.global_beta = global_beta

        self.global_indexing()
        self.assemble_M()
        self.assemble_K()

        return

    def global_indexing(self):

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

        return

    def assemble_M(self):

        front_board_M = self.front_board.M_matrix
        back_board_M = self.back_board.M_matrix
        bridge_M = self.bridge.M_matrix
        post_M = self.post.M_matrix
        string_M = self.string.M_matrix

        self.global_M = block_diag(front_board_M, back_board_M, bridge_M, post_M, string_M)

        return
    
    def assemble_K(self):
        
        def global_connection_matrix(
            global_dimension: int,
            board_global_coordinate_offset: int,
            connected_element_global_coordinate: int,
            board_element_tuple: CoordList,
            connected_element_locations: CoordList,
            spring_constant: float):

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

        front_board_K = self.front_board.K_matrix
        back_board_K = self.back_board.K_matrix
        bridge_K = self.bridge.K_matrix
        post_K = self.post.K_matrix
        string_K = self.string.K_matrix

        bridge_front_spring_constant = (self.front_board.k * self.bridge.K_bulk) / (self.front_board.k + self.bridge.K_bulk) / len(self.bridge.location_tuple)
        bridge_back_spring_constant = (self.back_board.k * self.bridge.K_bulk) / (self.back_board.k + self.bridge.K_bulk) / len(self.bridge.location_tuple)
        post_front_spring_constant = (self.front_board.k * self.post.K_bulk) / (self.front_board.k + self.post.K_bulk) / len(self.post.location_tuple)
        post_back_spring_constant = (self.back_board.k * self.post.K_bulk) / (self.back_board.k + self.post.K_bulk) / len(self.post.location_tuple)
        string_bridge_spring_constant = (self.string.eq_k * self.bridge.K_bulk) / (self.string.eq_k + self.bridge.K_bulk)

        bridge_front_conn = global_connection_matrix(
            self.global_size, self.front_board_index_offset, self.bridge_index_offset,
            self.front_board.interior_elements_tuple,
            self.bridge.location_tuple,
            bridge_front_spring_constant
        )

        bridge_back_conn = global_connection_matrix(
            self.global_size, self.back_board_index_offset, self.bridge_index_offset,
            self.back_board.interior_elements_tuple,
            self.bridge.location_tuple,
            bridge_back_spring_constant
        )

        post_front_conn = global_connection_matrix(
            self.global_size, self.front_board_index_offset, self.post_index_offset,
            self.front_board.interior_elements_tuple,
            self.post.location_tuple,
            post_front_spring_constant
        )

        post_back_conn = global_connection_matrix(
            self.global_size, self.back_board_index_offset, self.post_index_offset,
            self.back_board.interior_elements_tuple,
            self.post.location_tuple,
            post_back_spring_constant
        )

        string_bridge_conn = np.zeros((self.global_size, self.global_size))
        string_bridge_conn[self.string_index_offset, self.string_index_offset] += string_bridge_spring_constant
        string_bridge_conn[self.bridge_index_offset, self.bridge_index_offset] += string_bridge_spring_constant
        string_bridge_conn[self.string_index_offset, self.bridge_index_offset] -= string_bridge_spring_constant
        string_bridge_conn[self.bridge_index_offset, self.string_index_offset] -= string_bridge_spring_constant

        self.global_K = block_diag(front_board_K, back_board_K, bridge_K, post_K, string_K) + bridge_front_conn + bridge_back_conn + post_front_conn + post_back_conn + string_bridge_conn
        
        return


class VibrationSimulation:

    mode_shapes: FloatMatrix
    nat_freq: FloatMatrix

    input_type: int
    input_fractional_location: float
    input_amplitude: float

    sampling_rate: int
    simulation_period: float

    time_series_response: FloatMatrix

    time_sampling_points: FloatMatrix
    sampling_region_response: FloatMatrix

    ## FLAGS:
    BOW = 0
    TAP = 1
    PLUCK = 2

    def __init__(self,
        connected_system_obj: ConnectedSystem,
        input_type: int,
        input_fractional_location: float,
        input_amplitude: float,
        sampling_rate: int,
        simulation_period: float,
        acoustic_gain: float
        ) -> None:

        self.input_type = input_type
        self.input_fractional_location = input_fractional_location
        self.input_amplitude = input_amplitude
        self.sampling_rate = sampling_rate
        self.simulation_period = simulation_period

        global_K = connected_system_obj.global_K
        global_M = connected_system_obj.global_M
        alpha = connected_system_obj.global_alpha
        beta = connected_system_obj.global_beta

        self.nat_freq, self.mode_shapes = eigh(global_K, global_M)
        self.nat_freq = np.sqrt(self.nat_freq)

        self.state_response(
            global_K=global_K,
            global_M=global_M,
            alpha=alpha,
            beta=beta,
            global_size=connected_system_obj.global_size,
            num_string_node=connected_system_obj.string.num_node,
            string_index_offset=connected_system_obj.string_index_offset
        )
        self.soundwave_creation(
            sampling_zone=connected_system_obj.front_board.hole_elements_tuple,
            back_board_object=connected_system_obj.back_board,
            back_board_coordinate_offset=connected_system_obj.back_board_index_offset,
            gain=acoustic_gain
        )

        return

    def state_response(self, 
                            global_K: FloatMatrix, 
                            global_M: FloatMatrix,
                            alpha: float,
                            beta: float,
                            global_size: int,
                            num_string_node: int,
                            string_index_offset):

        time_step = 1/self.sampling_rate
        time_series = np.arange(0, self.simulation_period, time_step)
        diag_signal_data = np.empty((global_size,time_series.size))

        if self.input_type == self.BOW:
            pass
        
        elif self.input_type == self.TAP:
            pass

        elif self.input_type == self.PLUCK:
            local_input_node_index = int(self.input_fractional_location*num_string_node)
            slope_pre = (self.input_amplitude)/(local_input_node_index)
            slope_post = (-self.input_amplitude)/((num_string_node - 1) - (local_input_node_index + 1))
        
            string_local_init_vector_1st_half = np.array(range(local_input_node_index + 1))*slope_pre
            string_local_init_vector_2nd_half = np.array(range(0,num_string_node - local_input_node_index - 1))*(slope_post) + self.input_amplitude

            initial_condition_vector_local = np.concatenate((string_local_init_vector_1st_half,string_local_init_vector_2nd_half), axis=0)
            initial_condition_vector_global = np.zeros(global_size)
            initial_condition_vector_global[string_index_offset:(string_index_offset+num_string_node)] = initial_condition_vector_local
            diag_initial_condition_vector = np.linalg.inv(global_K) @ initial_condition_vector_global

            response_function_template = lambda t, x0, omega, zeta, omega_d: np.exp(-zeta*omega*t)*(x0*np.cos(omega_d*t) + zeta*omega*x0/omega_d*np.sin(omega_d*t))

            for mode in range(global_size):
                x0_j = diag_initial_condition_vector[mode]
                omega_j = self.nat_freq[mode]
                zeta_j = (alpha/omega_j + beta*omega_j)/2
                omega_d_j = omega_j * np.sqrt(1-zeta_j**2)
                diag_signal_data[mode,:] = response_function_template(time_series, x0=x0_j, omega=omega_j, zeta=zeta_j, omega_d=omega_d_j)
                # DEBUG
                print(f"Finished Calculating Mode {mode}")
            
            self.time_series_response = self.mode_shapes @ diag_signal_data
            self.time_sampling_points = time_series
        return

    def soundwave_creation(self, sampling_zone: CoordList, back_board_object: ConnectedSystem.BackBoard, back_board_coordinate_offset: int, gain: float):

        total_response = np.zeros(self.time_sampling_points.size)

        for coord in sampling_zone:
            local_index = back_board_object.interior_elements_tuple.index(coord)
            global_index = local_index + back_board_coordinate_offset
            element_response = self.time_series_response[global_index,:]
            total_response = total_response + element_response*gain
        
        self.sampling_region_response = total_response
        return
    
        


# Define Instrument
class Acoustic:

    global_system: ConnectedSystem
    system_response: VibrationSimulation

    def __init__(self, design: Geometry, material: Material, tuning: StringTuning, simulation: NoteSimulation):

        board_element_volume = design.board_thickness*design.meter_per_pixel**2 
        front_board_element_mass = material.front_board_density*board_element_volume
        back_board_element_mass = material.back_board_density*board_element_volume
        
        front_board = ConnectedSystem.FrontBoard(
            m = front_board_element_mass,
            k = material.front_board_k,
            k_diag = material.front_board_k_diag,
            wall_pixel_data = design.wall_profile,
            interior_pixel_data = design.board_profile,
            hole_pixel_data = design.holes_profile
        )
        print("Finished Creating Front Board")

        back_board = ConnectedSystem.BackBoard(
            m = back_board_element_mass,
            k = material.back_board_k,
            k_diag = material.back_board_k_diag,
            wall_pixel_data = design.wall_profile,
            interior_pixel_data = design.board_profile
        )
        print("Finished Creating Back Board")

        bridge = ConnectedSystem.Bridge(
            K_bulk = material.bridge_k,
            M_bulk = material.bridge_m,
            bridge_location_image = design.bridge_location
        )
        print("Finished Creating Bridge")

        post = ConnectedSystem.Post(
            K_bulk = material.post_k,
            M_bulk = material.post_m,
            post_location_image = design.post_location
        )
        print("Finished Creating Post")

        string = ConnectedSystem.String(
            string_tension = tuning.string_tension,
            string_mass_per_length = tuning.string_mass_per_length,
            eff_string_length = tuning.eff_string_length,
            num_node = tuning.num_node
        )
        print("Finished Creating String")

        self.global_system = ConnectedSystem(
            back_board = back_board,
            front_board = front_board,
            bridge = bridge,
            post = post,
            string = string,
            global_alpha = material.alpha,
            global_beta = material.beta
        )
        print("Finished Creating Global System")

        self.system_response = VibrationSimulation(
            connected_system_obj=self.global_system,
            input_type=VibrationSimulation.PLUCK,
            input_fractional_location=simulation.input_fractional_location,
            input_amplitude=simulation.input_amplitude,
            sampling_rate=simulation.sampling_rate,
            simulation_period=simulation.simulation_period,
            acoustic_gain=simulation.acoustic_gain
        )
        print("Finish Calculating System Response")

        return
    

if __name__ == "__main__":
    pass