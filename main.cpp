#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/opengl/MeshGL.h>
#include <igl/triangle_triangle_adjacency.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <igl/opengl/ViewerData.h>
#include <math.h>
#include <tuple>
#include <chrono>
#include "splinesurf.h"
#include <unordered_set>
#include <unordered_map>

// Global variables for color of edges and points
Eigen::RowVector3d edge_color = {1, 0, 0};
Eigen::RowVector3d point_color = {0, 1, 0};

// This function computes the global position of the spline control points and draws them with igl functionalities
void update_control_points_shape(splinesurf::Gui_Spline &s, const splinesurf::bezier_mesh &mesh, igl::opengl::glfw::Viewer &viewer)
{
    // igl wants two matrices to draw points:
    // - a matrix of positions P: an Nx3 matrix where each row describes a point for N points
    // - a matrix of colors C: an Nx3 matrix where each row defines in rgb the color of the point at the correspondent row of P
    Eigen::MatrixXd positions = {s.control_points.size(), 3};
    Eigen::MatrixXd colors = {s.control_points.size(), 3};

    // We are going to draw only anchor points, not tangents (hence i += 3).
    // for each control point we compute the global posiition starting from the mesh_point representation
    // and set the values in P and C
    for (int i = 0; i < positions.rows(); ++i)
    {
        positions.row(i) = eval_position(mesh.triangles, mesh.positions, s.control_points[i]);
        colors.row(i) = point_color;
    }
    // Call to the set_points function. This function overwrites all the points already plotted and draws only the points contained in P.
    viewer.data().set_points(positions, colors);
}

// This functions draws the spline and/or the control polygon
void update_tracing_structures(splinesurf::Gui_Spline &spline, const splinesurf::bezier_mesh &mesh, igl::opengl::glfw::Viewer &viewer, bool showControlPolygon){
    // Call to the set_edges function. This function overwrites all the lines already plotted and draws only the ones given.
    // The set_edges takes three inputs:
    // - A matrix of positions P: an Nx3 matrix where each row describes the position of a point
    // - A matrix of edges E: an Mx2 matrix of indices that describes which point is connected to which, using the P indices
    //   (if for example we have a line [0 1] inside E we are saying that the point at index 0 and the point at index 1 in P are connected)
    // - A matrix of colors C: an Mx3 matrix where each row describes the color in rgb of the correspondent point in P

    // If we want to draw the control polygon we concatenate the curve matrices and the control polygon one
    if(showControlPolygon){
        Eigen::MatrixXd positionsBezierAndCP = Eigen::MatrixXd(spline.positions.rows() + spline.cp_positions.rows(), 3);
        Eigen::MatrixXi edgesBezierAndCP = Eigen::MatrixXi(spline.edges.rows() + spline.cp_edges.rows(), 2);
        Eigen::MatrixXd colorsBezierAndCP = Eigen::MatrixXd(spline.colors.rows() + spline.cp_colors.rows(), 3);
        
        positionsBezierAndCP << spline.positions, spline.cp_positions;
        // control points edges indices have to be shifted of the number of rows of the spline positions matrix 
        // (the first control point will be in the matrix in the row after the last spline point)
        edgesBezierAndCP << spline.edges, spline.cp_edges.unaryExpr([spline](int old) { 
            int shift = spline.positions.rows();
            return old + shift;
        });
        colorsBezierAndCP << spline.colors, spline.cp_colors;

        viewer.data().set_edges(positionsBezierAndCP, edgesBezierAndCP, colorsBezierAndCP);
    }
    // If instead we want only the spline we plot it
    else
        viewer.data().set_edges(spline.positions, spline.edges, spline.colors);
}

// This function returns the mesh_point correspondent to the point clicked with the mouse on the mesh
splinesurf::mesh_point getRayTarget(igl::opengl::glfw::Viewer viewer, Eigen::MatrixXd &V, Eigen::MatrixXi &F){
    Eigen::Vector3d bc; // Barycentric coordinates of the clicked point
    int fid;            // Face id of the clicked point
    // Takes the x and y on the screen
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    // We use the igl unproject_onto_mesh function.
    // Given the position of the mouse on the screen it sets in fid and bc the face id and the barycentric coordinates
    // It returns true if the mouse intersects the mesh, false instead
    if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, fid, bc))
        return {fid, {bc[0], bc[1]}};
    // If no point is found an empty one is returned
    return {-1, Eigen::Vector2d{-1, -1}};
}

// This function takes in input the matrix of vertices V, the matrix of faces F and a mesh_point mp
// And returns the global position of the mesh_point
Eigen::RowVector3d fromBarycentricToSpace(Eigen::MatrixXd &V, Eigen::MatrixXi &F, splinesurf::mesh_point mp){
    Eigen::Vector3d v0 = V.row(F.row(mp.face)[0]);
    Eigen::Vector3d v1 = V.row(F.row(mp.face)[1]);
    Eigen::Vector3d v2 = V.row(F.row(mp.face)[2]);
    return (v0 * mp.uv[0] + v1 * mp.uv[1] + v2 * (1 - mp.uv[0] - mp.uv[1])).transpose();
}

// This function adds the given mesh_point mp to the control_points vector in the given spline
void add_control_point(splinesurf::Gui_Spline &s, splinesurf::mesh_point mp){
    s.control_points.push_back(mp);
}

// This function adds a new curve to the spline
void add_new_curve_to_spline(splinesurf::Gui_Spline &s, splinesurf::bezier_mesh m){
    int curve_id = (int)s.curves.size();
    s.curves.push_back({});
    // here in the original version was the tangent computation
    // now it is not needed since we feed manually all the control polygon points
    s.curves_to_update.insert(curve_id);
}

int main(int argc, char *argv[]){
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd spline_points(0, 3);
    Eigen::MatrixXi spline_edges(0, 2);
    Eigen::MatrixXd points_colors(0, 3);
    Eigen::MatrixXd edges_colors(0, 3);

    std::vector<splinesurf::mesh_point> control_points;
    std::unordered_set<int> curves_to_update;
    std::vector<splinesurf::Gui_Curve> curves;
    Eigen::MatrixXd positions(0, 3);
    Eigen::MatrixXi edges(0, 2);
    Eigen::MatrixXd colors(0, 3);
    Eigen::MatrixXd cp_positions(0, 3);
    Eigen::MatrixXi cp_edges(0, 2);
    Eigen::MatrixXd cp_colors(0, 3);

    splinesurf::Gui_Spline spline = {
        control_points,
        curves_to_update,
        curves,
        positions,
        edges,
        colors,
        cp_positions,
        cp_edges,
        cp_colors
    };

    splinesurf::bezier_params params;

    // Load a mesh in OFF format
    if (argc == 1){
        igl::readOBJ("../data/bunny-small.obj", V, F);
    } else{
        std::cout << "filename: " << argv[1] << std::endl;
        if(!igl::readOBJ(argv[1], V, F)){
            std::cout << "ERROR: " << argv[1] << " not found" << std::endl;
            return -1;
        }
    }
    Eigen::MatrixXi TT, TTi;
    igl::triangle_triangle_adjacency(F, TT, TTi);
    splinesurf::dual_geodesic_solver dual_solver = splinesurf::make_dual_geodesic_solver(F, V, TT);
    splinesurf::bezier_mesh mesh{F, TT, V, dual_solver};

    // Init the viewer
    igl::opengl::glfw::Viewer viewer;
    viewer.core().is_animating = true;
    viewer.data().point_size = 10;
    viewer.data().line_width = 3.0f;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);
    static bool showControlPolygon = false;

    // Customize the menu
    double doubleVariable = 0.1f; // Shared between two menus

    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&](){
        // Draw parent menu content
        menu.draw_viewer_menu();
        
        // add control polygon checkbox
        ImGui::Checkbox("Show control polygon", &showControlPolygon);

        // add control polygon button
        if (ImGui::Button("Toggle Control Polygon", ImVec2(-1,0))){
            showControlPolygon = !showControlPolygon;
            if(spline.curves.size() >= 1)
                update_tracing_structures(spline, mesh, viewer, showControlPolygon);
        }
    };

    int added_points = 0;
    bool first_curve = true;    // first needs 4 points, others only 3
    bool moved = false;     // drag detection for mesh rotation
    splinesurf::mesh_point cp_to_be_added;
    
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer &viewer, int, int) -> bool{
        // cast a ray from screen to scene and get the corresponding mesh point
        splinesurf::mesh_point target = getRayTarget(viewer, V, F);
        if (target.face != -1){     // if ray hit a point on the mesh
            moved = false;
            Eigen::RowVector3d ray_target = fromBarycentricToSpace(V, F, target);   // get target point in world coords
            cp_to_be_added = target;    // prepare point to be added to control points
            return false;
        }
        return false;
    };

    viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer &viewer, int, int) -> bool{
        moved = true;       // if dragging inhibits the control point add in callback_mouse_up
        return false;
    };

    viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer &viewer, int, int) -> bool{
        if (cp_to_be_added.face != -1 && !moved){       // if current point comes from a successful ray cast on mesh and if the mouse did not move after click
            add_control_point(spline, cp_to_be_added);
            update_control_points_shape(spline, mesh, viewer);      // updates igl drawing structure for points
            added_points++;
            if (first_curve && added_points == 4){
                add_new_curve_to_spline(spline, mesh);      // add the first bezier curve
                first_curve = false;    
                added_points = 0;       // reset counter
            }
            else if (!first_curve && added_points == 3){
                add_new_curve_to_spline(spline, mesh);      // add a new bezier curve
                added_points = 0;   // reset counter
            }
            cp_to_be_added = {-1, Eigen::Vector2d(-1,-1)};      // clear current point
        }
        moved = false;
        return false;
    };

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool{
        for (auto &k : spline.curves_to_update){
            update_curve_shape(spline, k, mesh, params);    // generate points for kth curve
            update_tracing_structures(spline, mesh, viewer, showControlPolygon);
        }
        spline.curves_to_update.clear();
        return false;
    };

    // Plot the mesh
    viewer.data().set_mesh(V, F);
    viewer.data().add_label(viewer.data().V.row(0) + viewer.data().V_normals.row(0).normalized() * 0.005, "Splinesurf");
    viewer.launch();
}