from raytracing import geometry as geom
from raytracing import plot_utils
from raytracing import file_utils


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import json



class RayTracingProblem:
    """
    A ray tracing problem instance offers tools to find paths between a given emitter and receivers.
    A path may be direct or indirect, as a result of reflection and/or diffraction.

    The paths are found regardless of the type of rays (sound wave, E.M. wave, light wave, ...).

    :param emitter: the emitter point
    :type emitter: numpy.array *size=(3)*
    :param place: the place of the problem, which should contain buildings and a ground surface
    :type place: raytracing.OrientedPlace
    :param receivers: the receiver points but will also take the points contained in the place as receivers
    :type receivers: numpy.array *shape=(N, 3)*
    :param n_screens: the number of screens through which the emitter will emit rays,
        by default takes all the screens / faces (6) of the cube encapsulating the emitter
        but if n_screens < 6, then will take the n_screens screens witch see the most
        polygons (i.e.: through each screen, the emitter can see a given amount of polygons
        and the goal is to avoid losing any possible polygon on which reflection is possible)
    :type n_screens: int, 0 < n_screens <= 6
    """
    def __init__(self, emitter, place, receivers=None, n_screens=6):
        assert 0 < n_screens <= 6
        self.emitter = emitter
        self.n_screens = n_screens
        self.place = place
        self.polygons=None
        self.receivers=np.array([])
        if self.place is not None:
            self.polygons = np.array(place.get_polygons_list(),dtype=object)
            self.receivers = self.place.set_of_points
        if receivers is not None:
            self.place.add_set_of_points(receivers)    
        self.solved_receivers=None
        self.visibility_matrix = None
        self.distance_to_screen = None
        self.emitter_visibility = None
        self.sharp_edges = None
        n = self.receivers.shape[0]
        self.los = {
            r: list()  # Lines of sight
            for r in range(n)
        }
        self.reflections = {
            r: defaultdict(list)
            for r in range(n)
        }
        self.diffractions = {
            r: defaultdict(list)
            for r in range(n)
        }
        if self.place is not None:
            self.precompute()

    def precompute(self):
        """
        Pre-computes order-of-path independent results such as:
        - visibility matrices
        - sharp edges
        - lines of sight (a.k.a. direct paths)
        """
        self.visibility_matrix = self.place.get_visibility_matrix(strict=False)
        cube = geom.Cube.by_point_and_side_length(self.emitter, 2 * 0.1)

        self.distance_to_screen = cube.polygons[0].distance_to_point(self.emitter)

        emitter_visibilities = list()

        for i in range(self.n_screens):
            emitter_visibility = geom.polygon_visibility_vector(
                cube.polygons[i], self.polygons, strict=False
            )
            visibility = np.sum(emitter_visibility)
            emitter_visibilities.append((visibility, emitter_visibility))

        emitter_visibilities.sort(key=lambda x: x[0], reverse=True)

        self.emitter_visibility = emitter_visibilities[0][1]

        for _, emitter_visibility in emitter_visibilities[1:self.n_screens]:
            self.emitter_visibility |= emitter_visibility

        self.sharp_edges = self.place.get_sharp_edges()

        visible_polygons = self.polygons[self.emitter_visibility]

        for r, receiver in enumerate(self.receivers):
            line = np.row_stack([self.emitter, receiver])
            if not geom.polygons_obstruct_line_path(visible_polygons, line):
                self.los[r].append(line)

    def get_visible_polygons_indices(self, index):
        indices = self.visibility_matrix[index, :]
        return np.where(indices)[0]

    def check_reflections(self, lines, polygons_indices):
        """
        Checks whether a reflection path is valid. It is valid if it satisfies 2 conditions:
        1. Each point is contained in the polygon it should be in
        2. No polygon obstructs the line path

        :return: True if reflection path is valid
        :rtype: bool
        """
        for i, index in enumerate(polygons_indices):
            if not self.polygons[index].contains_point(lines[i + 1, :], check_in_plane=True):
                return False

        if geom.polygons_obstruct_line_path(self.polygons, lines[:2, :]):
            return False

        for i, _ in enumerate(polygons_indices):
            if geom.polygons_obstruct_line_path(self.polygons, lines[i + 1:i + 3, :]):
                return False

        return True

    def check_reflections_and_diffraction(self, lines, polygons_indices, edge):
        return self.check_reflections(lines[:-1, :], polygons_indices) and \
               geom.point_on_edge_(lines[-2, :], edge) and \
               not geom.polygons_obstruct_line_path(self.polygons, lines[-2:, :])

    def solve(self, max_order=2,receivers_indexs=None):
        emitter = self.emitter
        #receivers = self.receivers
        indices = np.where(self.emitter_visibility)[0]

     
        if receivers_indexs is not None:
            #solve only specified receivers
            assert all(i < len(self.receivers) for i in receivers_indexs) ,f"provided RX indexs {receivers_indexs} but only {len(self.receivers)} RX exist"
            receivers=[]
            for index in receivers_indexs:
                receivers.append(self.receivers[index])
        else: 
            #solve all receivers
            receivers=self.receivers   
        print(f"solving for the following receivers {receivers}")

        # Only reflections
        def recursive_reflections(polygons_indices, order):
            planes_parametric = [self.polygons[index].get_parametric() for index in polygons_indices]
            idx=0
            for r, receiver in enumerate(receivers):
                if receivers_indexs is not None:
                    r=receivers_indexs[idx]
                    idx+=1
                
                points, sol = geom.reflexion_points_from_origin_destination_and_planes(emitter, receiver, planes_parametric)

                if not sol.success:
                    continue

                lines = np.row_stack([emitter, points, receiver])
                if self.check_reflections(lines, polygons_indices):
                    self.reflections[r][order].append((lines, polygons_indices))

            if order == max_order:
                return
            else:
                index = polygons_indices[-1]
                indices = self.get_visible_polygons_indices(index)
                for i in indices:
                    recursive_reflections(polygons_indices + [i], order=order + 1)

        if max_order >= 1:
            print('Iterating through all n reflect.')
            for index in tqdm(indices):
                recursive_reflections([index], 1)

        if max_order < 1:
            return

        # Reflections and 1 diffraction
        print('Iterating through all 1 diff.')
        for (i, j), edge in tqdm(self.sharp_edges.items()):
            if self.emitter_visibility[i] or self.emitter_visibility[j]:
                idx=0
                for r, receiver in enumerate(receivers):
                    if receivers_indexs is not None:
                        r=receivers_indexs[idx]
                        idx+=1
                    points, sol = geom.reflexion_points_and_diffraction_point_from_origin_destination_planes_and_edge(
                        emitter, receiver, [], edge
                    )

                    if not sol.success:
                        continue

                    lines = np.row_stack([emitter, points, receiver])
                    if self.check_reflections_and_diffraction(lines, [], edge):
                        self.diffractions[r][1].append((lines, [], (i, j), edge))

        if max_order < 2:
            return

        def recursive_reflections_and_diffraction(polygons_indices, order):
            planes_parametric = [self.polygons[index].get_parametric() for index in polygons_indices]

            last_index = polygons_indices[-1]

            visible_polygons_indices = self.get_visible_polygons_indices(last_index)

            for edge_polygons, edge in self.sharp_edges[(*visible_polygons_indices, ...)]:
                idx=0
                for r, receiver in enumerate(receivers):  
                    if receivers_indexs is not None:
                        r=receivers_indexs[idx]
                        idx+=1
                    
                    points, sol = geom.reflexion_points_and_diffraction_point_from_origin_destination_planes_and_edge(emitter, receiver, planes_parametric, edge)

                    if not sol.success:
                        continue

                    lines = np.row_stack([emitter, points, receiver])
                    if self.check_reflections_and_diffraction(lines, polygons_indices, edge):
                        self.diffractions[r][order].append((lines, polygons_indices, edge_polygons, edge))

            if order == max_order:
                return
            else:
                index = polygons_indices[-1]
                indices = self.get_visible_polygons_indices(index)
                for i in indices:
                    recursive_reflections_and_diffraction(polygons_indices + [i], order=order + 1)

        print('Iterating through all n-1 reflect. and 1 diff.')
        for index in tqdm(indices):
            recursive_reflections_and_diffraction([index], 2)
            
        self.solved_receivers=receivers    

    def save(self, filename): 
        data = {
            'place': self.place.to_json(),
            'solved_receivers':self.solved_receivers,
            'emitter': self.emitter,
            'los': self.los,
            'reflections': self.reflections,
            'diffractions': self.diffractions
        }
        #save in a readable json, loadable.
        filename=filename if ".json" in filename else f'{filename}.json'
        file_utils.json_save(filename, data, cls=geom.OrientedGeometryEncoder)   
        return
            
    def load(self,filename):  
        assert 1==2,"loading a problem from a json is not working"
        #TODO load correctly, this does not work.
        reader=open(filename,"r")
        dictionnary=json.load(reader)         
        reader.close()
        #dictionnary=file_utils.json_load(filename)
        self.solved_receivers=np.array(dictionnary["solved_receivers"])
        self.emitter=np.array(dictionnary["emitter"])
        place=geom.OrientedPlace.from_json(data=dictionnary["place"])
        self.place=place
         
        #convert indices from strings to ints
        los={int(key):dictionnary['los'][key] for key in dictionnary['los']}
        #convert lists to ndarrays
        for receiver in range(0,len(los)):
            if los[receiver]!=[]:
                los[receiver][0]=np.array(los[receiver][0])
        self.los=los
        
        reflections={int(key):dictionnary['reflections'][key] for key in dictionnary['reflections']}
        for receiver in range(0,len(reflections)):
            reflections[receiver]={int(key):reflections[receiver][key] for key in reflections[receiver]}
            reflections[receiver]=defaultdict(list,reflections[receiver])
            for order in range(1,len(reflections[receiver])+1):
                for ref in range(0,len(reflections[receiver][order])):
                    reflections[receiver][order][ref]=(np.array(reflections[receiver][order][ref][0]),reflections[receiver][order][ref][1])

        self.reflections=reflections
        diffractions={int(key):dictionnary['diffractions'][key] for key in dictionnary['diffractions']}
        for receiver in range(0,len(diffractions)):
            diffractions[receiver]={int(key):diffractions[receiver][key] for key in diffractions[receiver]}
            diffractions[receiver]=defaultdict(list,diffractions[receiver])
            for order in range(1,len(diffractions[receiver])+1):
                for diff in range(0,len(diffractions[receiver][order])):
                    diffractions[receiver][order][diff]=(np.array(diffractions[receiver][order][diff][0]),
                                                          diffractions[receiver][order][diff][1],
                                                          tuple(diffractions[receiver][order][diff][2]),
                                                          np.array(diffractions[receiver][order][diff][3])
                                                          )
        self.diffractions=diffractions
        self.polygons = np.array(place.get_polygons_list(),dtype=object)
        self.n_screens=6 #TODO load correctly 
        #self.precompute() #compute the other arguments (visibility matrix etc)
        
        return self
        
        
        
    def plot3d(self, ax=None, ret=False, show_refl=True, show_diff=True, receivers_indexs=None,legend=False):
        """
        plot a single 3d plot with the ray paths

        """
        ax = plot_utils.get_3d_plot_ax(ax)
        self.place.plot3d(ax=ax, points_kwargs=dict(color='k', s=20))
        plot_utils.add_points_to_3d_ax(ax, self.emitter, color='r', s=20)
        plot_utils.add_text_at_point_3d_ax(ax, self.emitter, 'TX')

        first = True
        handles = []
        labels = []
        
        
        if receivers_indexs is not None:
            receivers_to_plot=iter(list(receivers_indexs))
        else:
            receivers_to_plot=iter(list(range(len(self.receivers))))
        
        for r in receivers_to_plot:
            for line in self.los[r]:
                line3D, = plot_utils.add_line_to_3d_ax(ax, line, color='b')
                if first:
                    handles.append(line3D)
                    labels.append('LOS')
                    first = False

            colors = {
                1: 'g',
                2: 'm',
                3: 'y',
                4: 'r'
            }

            if show_refl:
                for order, lines in self.reflections[r].items():
                    first = True
                    color = colors[order]
                    for line, _ in lines:
                        line3D, = plot_utils.add_line_to_3d_ax(ax, line, color=color)
                        plot_utils.add_points_to_3d_ax(ax, line[1:order+1, :], color=color)

                        if first:
                            handles.append(line3D)
                            labels.append(f'{order} reflect.')
                            first = False

            if show_diff:
                for order, lines in self.diffractions[r].items():
                    first = True
                    color = colors[order]
                    for line, _, _, _ in lines:
                        line3D, = plot_utils.add_line_to_3d_ax(ax, line, color=color, linestyle='--')
                        plot_utils.add_points_to_3d_ax(ax, line[1:order+1, :], color=color)

                        if first:
                            handles.append(line3D)
                            labels.append(f'{order-1} reflect. and 1 diff.')
                            first = False
            
        self.place.center_3d_plot(ax)
        if legend:
            ax.legend(handles, labels)
        if ret:
            return ax


    
    def plot_all_rays(self):
        """
        plot the ray paths for each receivers in a different subplot
        """
        MAXPLOTS=4
        fig = plt.figure("Ray traced places",figsize=(8,5))
        fig.set_dpi(300)
        nplots=len(self.solved_receivers)
        if nplots>MAXPLOTS:
            print(f"won't plot {nplots}: it will be unreadable, plotting {MAXPLOTS} firsts instead")
            nrows,ncols=plot_utils.get_subplot_row_columns(MAXPLOTS)  
            for i in range(MAXPLOTS):
                ax = fig.add_subplot(nrows, ncols, i+1, projection = '3d') #total rows,total columns, index
                ax = self.plot3d(ax=ax,receivers_indexs=[i],ret=True,legend=True)
                ax.set_title('RX'+str(i))
                plt.show(block=False)
                plt.pause(0.001) 
            
        else:
            nrows,ncols=plot_utils.get_subplot_row_columns(nplots)   
            for i in range(nplots):
                ax = fig.add_subplot(nrows, ncols, i+1, projection = '3d') #total rows,total columns, index
                ax = self.plot3d(ax=ax,receivers_indexs=[i],ret=True)
                ax.set_title('RX'+str(i))
               
            plt.show(block=False)
            plt.pause(0.001) 
        
        return
        
        
        
        
    
