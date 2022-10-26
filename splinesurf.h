#include <vector>
#include <unordered_set>
#include <Eigen/Dense>
#include <Eigen/Sparse>

inline const auto flt_max = std::numeric_limits<float>::max();

namespace splinesurf {

    struct mesh_point {
        int face = -1;
        Eigen::Vector2d uv = {0, 0};
    };

    struct mesh_path {
        std::vector<mesh_point> points;
    };

    struct geodesic_path {
        mesh_point start = {};
        mesh_point end = {};
        std::vector<int> strip = {};
        std::vector<float> lerps = {};
    };

    struct dual_geodesic_solver {
        struct edge {
            int node = -1;
            float length = flt_max;
        };
        std::vector<std::array<edge, 3>> graph = {};
        std::vector<int> parent_faces = {};
        int num_original_faces = 0;
    };

    struct bezier_mesh {
        Eigen::MatrixXi triangles = {};
        Eigen::MatrixXi adjacencies = {};
        Eigen::MatrixXd positions = {};
        dual_geodesic_solver dual_solver = {};
    };

    struct bezier_node {
        std::array<mesh_point, 4> points = {};
        std::array<geodesic_path, 3> lines = {};
        int parent = -1;
        int children[2] = {-1, -1};
        float t_start = 0;
        float t_end = 1;
    };

    struct bezier_tree {
        std::vector<bezier_node> nodes = {};
        int depth = 0;
    };

    struct Gui_Line {
        geodesic_path path = {};
        Eigen::MatrixXd positions = {};
        float length = 0;
    };

    struct Gui_Curve {
        Eigen::MatrixXd positions = {};
        std::array<Gui_Line, 2> tangents = {};
    };

    struct Gui_Spline {
        std::vector<mesh_point> control_points = {};
        std::unordered_set<int> curves_to_update = {};
        std::vector<Gui_Curve> curves = {};
        // structures for drawing beziers and control polygons in the igl viewer
        Eigen::MatrixXd positions = {};
        Eigen::MatrixXi edges = {};
        Eigen::MatrixXd colors = {};
        Eigen::MatrixXd cp_positions = {};
        Eigen::MatrixXi cp_edges = {};
        Eigen::MatrixXd cp_colors = {};
    };

    enum struct spline_algorithm {
        de_casteljau_uniform = 0,
        de_casteljau_adaptive,
        de_casteljau_classic,
        subdivision_uniform,
        subdivision_adaptive,
        karcher,
        flipout
    };
    
    const auto spline_algorithm_names = std::vector<std::string>{
        "de_casteljau_uniform", "de_casteljau_adaptive", "de_casteljau_classic",
        "subdivision_uniform", "subdivision_adaptive", "karcher", "flipout"
    };

    struct bezier_params {
        spline_algorithm algorithm = spline_algorithm::de_casteljau_uniform;
        int subdivisions = 4;
        float precision = 0.1;
        float min_curve_size = 0.001;
        int  max_depth = 10;
        bool parallel = false;
    };

    struct funnel_point {
        int face = 0;
        Eigen::Vector2d pos  = {0, 0};
    };

    // === <DEBUG UTILITIES> ===
    template <typename T>
    void printVector(T v);
    void printMeshPoint(mesh_point p);
    void printGeodesicPath(geodesic_path g);
    void printGuiLine(Gui_Line l);
    void printCurve(Gui_Curve curve);
    void printGuiSpline(Gui_Spline s);
    void printDualGeodesicSolver(dual_geodesic_solver s);
    // === </DEBUG UTILITIES> ===

    using unfold_triangle = std::array<Eigen::Vector2d, 3>;
    using bezier_segment = std::array<mesh_point, 4>;

    Eigen::Vector3d eval_position(const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions, const mesh_point &sample);
    void update_curve_shape(Gui_Spline& spline, const int curve_id, const bezier_mesh& mesh, const bezier_params& params);
    dual_geodesic_solver make_dual_geodesic_solver(const Eigen::MatrixXi &_triangles, const Eigen::MatrixXd &_positions, const Eigen::MatrixXi &_adjacencies);
}