#include "splinesurf.h"
#include <math.h>
#include <deque>
#include <unordered_map>
#include <igl/barycentric_coordinates.h>
#include <iostream>

namespace splinesurf{

    // === <DEBUG UTILITIES> ===

    template <typename T>
    void printVector(T v){
        std::cout << "[";
        for(int i=0; i<v.size(); ++i){
            std::cout << v[i];
            if(i < v.size()-1)
                std::cout << ", ";
        }
        std::cout << "]";
    }

    void printMeshPoint(mesh_point p){
        std::cout << " - face: " << p.face << std::endl;
        std::cout << " - uv: ";
        printVector(p.uv);
        std::cout << std::endl;
    }

    void printGeodesicPath(geodesic_path g){
        std::cout << " - start: ";
        printMeshPoint(g.start);
        std::cout << std::endl;
        std::cout << " - end: ";
        printMeshPoint(g.end);
        std::cout << std::endl;
        std::cout << " - strip: ";
        printVector(g.strip);
        std::cout << std::endl;
        std::cout << " - lerps: ";
        printVector(g.lerps);
        std::cout << std::endl;
    }

    void printGuiLine(Gui_Line l){
        std::cout << "LINE" << std::endl;
        std::cout << " - length: " << l.length << std::endl;
        std::cout << " - path: ";
        printGeodesicPath(l.path);
        std::cout << std::endl;
        std::cout << " - positions: ";
        std::cout << l.positions << std::endl;
        std::cout << std::endl;
    }

    void printCurve(Gui_Curve curve){
        std::cout << "CURVE" << std::endl;
        std::cout << " - positions: ";
        std::cout << curve.positions << std::endl;
        std::cout << std::endl;
        std::array<Gui_Line, 2> tangents = {};
        std::cout << " - tangent[0]: " << std::endl;
        printGuiLine(curve.tangents[0]);
        std::cout << std::endl;
        std::cout << " - tangent[1]: " << std::endl;
        printGuiLine(curve.tangents[1]);
    }

    void printGuiSpline(Gui_Spline s){
        std::cout << "SPLINE" << std::endl;
        std::cout << " - control points:" << std::endl;
        for(int i=0; i<s.control_points.size(); ++i){
            printMeshPoint(s.control_points[i]);
            if(i < s.control_points.size()-1)
                std::cout << ", ";
        }
        std::cout << std::endl << " - curves:" << std::endl;
        for(int i=0; i<s.curves.size(); ++i){
            printCurve(s.curves[i]);
            if(i < s.curves.size()-1)
                std::cout << ", ";
        }
        std::cout << std::endl;
    }

    void printDualGeodesicSolver(dual_geodesic_solver s){
        for(int i=0; i<s.graph.size(); ++i){
            std::cout << "[";
            for(int j=0; j<3; ++j)
                std::cout << "(node = " << s.graph[i][j].node << ", len = " << s.graph[i][j].length << "), ";
            std::cout << "]" << std::endl;
        }
    }

    // === </DEBUG UTILITIES> ===

    Eigen::Vector3d lerp(const Eigen::Vector3d& a, const Eigen::Vector3d& b, float u) {
        return a * (1 - u) + b * u;
    }

    Eigen::Vector2d lerp(const Eigen::Vector2d& a, const Eigen::Vector2d& b, float u) {
        return a * (1 - u) + b * u;
    }

    int mod3(int i) { 
        if(i > 2)
            return i - 3;
        return i; 
    }
    
    float cross(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return a[0] * b[1] - a[1] * b[0];
    }

    template <typename Out>
    Out interpolate_triangle(const Out p0, const Out p1, const Out &p2, const Eigen::Vector2d& uv) {
        return p0 * uv[0] + p1 * uv[1] + p2 * (1 - uv[0] - uv[1]);
    }

    Eigen::Vector3d eval_position(const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions, const mesh_point &sample){
        Eigen::Vector3i face_verts = Eigen::Vector3i(triangles.row(sample.face));
        return interpolate_triangle(Eigen::Vector3d(positions.row(face_verts[0])), Eigen::Vector3d(positions.row(face_verts[1])), Eigen::Vector3d(positions.row(face_verts[2])), sample.uv);
    }

    Eigen::Vector3d eval_position(const bezier_mesh& mesh, const mesh_point& point) {
       return eval_position(mesh.triangles, mesh.positions, point);
    }

    template <typename Update, typename Stop, typename Exit> 
    void search_strip(std::vector<float>& field, std::vector<bool>& in_queue, const dual_geodesic_solver& solver, const Eigen::MatrixXi& triangles, const Eigen::MatrixXd& positions, int start, int end, Update&& update, Stop&& stop, Exit&& exit) {
        auto destination_pos = eval_position(triangles, positions, {end, {1.0f / 3, 1.0f / 3}});
        auto estimate_dist = [&](int face) {
            auto p = eval_position(triangles, positions, {face, {1.0f / 3, 1.0f / 3}});
            return (p - destination_pos).norm();
        };
        field[start] = estimate_dist(start);

        // Cumulative weights of elements in queue. Used to keep track of the
        // average weight of the queue.
        double cumulative_weight = 0.0;

        // setup queue
        auto queue = std::deque<int>{};
        in_queue[start] = true;
        cumulative_weight += field[start];
        queue.push_back(start);

        while (!queue.empty()) {
            auto node = queue.front();
            auto average_weight = (float)(cumulative_weight / queue.size());

            // Large Label Last (see comment at the beginning)
            for (auto tries = 0; tries < queue.size() + 1; tries++) {
                if (field[node] <= average_weight) 
                    break;
                queue.pop_front();
                queue.push_back(node);
                node = queue.front();
            }

            // Remove node from queue.
            queue.pop_front();
            in_queue[node] = false;
            cumulative_weight -= field[node];

            // Check early exit condition.
            if (exit(node)) 
                break;
            if (stop(node)) 
                continue;

            for (auto i = 0; i < (int)solver.graph[node].size(); i++) {
                auto neighbor = solver.graph[node][i].node;
                if (neighbor == -1) 
                    continue;

                // Distance of neighbor through this node
                auto new_distance = field[node];
                new_distance += solver.graph[node][i].length;
                new_distance += estimate_dist(neighbor);
                new_distance -= estimate_dist(node);

                auto old_distance = field[neighbor];
                if (new_distance >= old_distance) 
                    continue;

                if (in_queue[neighbor]) {
                    // If neighbor already in queue, don't add it.
                    // Just update cumulative weight.
                    cumulative_weight += new_distance - old_distance;
                } 
                else {
                    // If neighbor not in queue, add node to queue using Small Label
                    // First (see comment at the beginning).
                    if (queue.empty() || (new_distance < field[queue.front()]))
                        queue.push_front(neighbor);
                    else
                        queue.push_back(neighbor);

                    // Update queue information.
                    in_queue[neighbor] = true;
                    cumulative_weight += new_distance;
                }
                // Update distance of neighbor.
                field[neighbor] = new_distance;
                if (update(node, neighbor, new_distance)) 
                    return;
            }
        }
    }

    std::vector<int> compute_strip_tlv(const bezier_mesh& mesh, int start, int end) {
        if (start == end) 
            return {start};

        thread_local static auto parents = std::vector<int>{};
        thread_local static auto field = std::vector<float>{};
        thread_local static auto in_queue = std::vector<bool>{};

        if (parents.size() != mesh.dual_solver.graph.size()) {
            parents.assign(mesh.dual_solver.graph.size(), -1);
            field.assign(mesh.dual_solver.graph.size(), flt_max);
            in_queue.assign(mesh.dual_solver.graph.size(), false);
        }

        // initialize once for all and sparsely cleanup at the end of every solve
        auto visited = std::vector<int>{start};
        auto sources = std::vector<int>{start};
        auto update  = [&visited, end](int node, int neighbor, float new_distance) {
            parents[neighbor] = node;
            visited.push_back(neighbor);
            return neighbor == end;
        };
        auto stop = [](int node) { return false; };
        auto exit = [](int node) { return false; };

        search_strip(field, in_queue, mesh.dual_solver, mesh.triangles, mesh.positions, start, end, update, stop, exit);

        // extract_strip
        auto strip = std::vector<int>{};
        auto node  = end;
        strip.reserve((int)sqrt(parents.size()));
        while (node != -1) {
            strip.push_back(node);
            node = parents[node];
        }

        // cleanup buffers
        for (auto& v : visited) {
            parents[v]  = -1;
            field[v]    = flt_max;
            in_queue[v] = false;
        }
        return strip;
    }

    Eigen::Vector2d intersect_circles(const Eigen::Vector2d &c2, double R2, const Eigen::Vector2d &c1, double R1) {
        double R = (c2 - c1).squaredNorm();
        double invR = 1 / R;
        Eigen::Vector2d result = (c1 + c2);
        result =  result + (c2 - c1) * ((R1 - R2) * invR);
        double A = 2 * (R1 + R2) * invR;
        double B = (R1 - R2) * invR;
        double s = A - B * B - 1;
        assert(s >= 0);
        result += Eigen::Vector2d(c2[1] - c1[1], c1[0] - c2[0]) * sqrt(s);
        // result = result * sqrt(s);
        return result / 2;
    }

    // assign 2D coordinates to vertices of the triangle containing the mesh
    // point, putting the point at (0, 0)
    unfold_triangle triangle_coordinates(const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions, const mesh_point &point) {
        auto first = unfold_triangle{};
        Eigen::RowVector3i tr = triangles.row(point.face);
        first[0] = {0, 0};
        first[1] = {0, (positions.row(tr[0]) - positions.row(tr[1])).norm()};
        auto rx = (positions.row(tr[0]) - positions.row(tr[2])).squaredNorm();
        auto ry = (positions.row(tr[1]) - positions.row(tr[2])).squaredNorm();
        first[2] = intersect_circles(first[0], rx, first[1], ry);

        // Transform coordinates such that point = (0, 0)
        auto point_coords = interpolate_triangle(first[0], first[1], first[2], point.uv);
        first[0] -= point_coords;
        first[1] -= point_coords;
        first[2] -= point_coords;
        return first;
    }

    // find a value in a vector or vecs
    static int find_in_vec(const std::vector<int> &vec, int x) {
        for (int i = 0; i < vec.size(); i++)
            if (vec[i] == x) 
                return i;
        return -1;
    }

    // find a value in a vector or vecs
    static int find_in_vec(const Eigen::Vector3i &vec, int x) {
        for (int i = 0; i < vec.rows(); i++){
            if (vec[i] == x)
                return i;
        }
        return -1;
    }

    inline int find_adjacent_triangle(const Eigen::Vector3i &triangle, const Eigen::Vector3i &adjacent) {
        for (int i = 0; i < 3; i++) { 
            auto k = find_in_vec(adjacent, triangle[i]);
            if (k != -1) {
                if (find_in_vec(adjacent, triangle[mod3(i + 1)]) != -1)
                    return i;
                else
                    return mod3(i + 2);
            }
        }
        assert(0 && "input triangles are not adjacent");
        return -1;
    }

    int find_adjacent_triangle(const Eigen::MatrixXi &triangles, int face, int neighbor) {
        return find_adjacent_triangle(triangles.row(face), triangles.row(neighbor));
    }

    // given the 2D coordinates in tanget space of a triangle, find the coordinates
    // of the k-th neighbor triangle
    unfold_triangle unfold_face(const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions, const unfold_triangle &tr, int face, int neighbor) {
        auto k = find_adjacent_triangle(triangles, face, neighbor);
        auto j = find_adjacent_triangle(triangles, neighbor, face);
        assert(j != -1);
        assert(k != -1);
        auto v = triangles.row(neighbor)[mod3(j + 2)];
        auto a = triangles.row(face)[k];
        auto b = triangles.row(face)[mod3(k + 1)];
        auto r0 = (positions.row(v) - positions.row(a)).squaredNorm();
        auto r1 = (positions.row(v) - positions.row(b)).squaredNorm();

        auto res = unfold_triangle{};
        res[j] = tr[mod3(k + 1)];
        res[mod3(j+1)] = tr[k];
        res[mod3(j+2)] = intersect_circles(res[j], r1, res[(j+1) % 3], r0);
        return res;
    }

    // assign 2D coordinates to a strip of triangles. point start is at (0, 0)
    std::vector<unfold_triangle> unfold_strip(const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions, const std::vector<int> &strip, const mesh_point &start) {
        auto coords = std::vector<unfold_triangle>(strip.size());
        coords[0] = triangle_coordinates(triangles, positions, start);
        for (auto i = 1; i < strip.size(); i++) {
            coords[i] = unfold_face(triangles, positions, coords[i - 1], strip[i - 1], strip[i]);
        }

        return coords;
    }

    // Create sequence of 2D segments (portals) needed for funneling.
    static std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> make_funnel_portals(const Eigen::MatrixXi &triangles, const std::vector<unfold_triangle> &coords, const std::vector<int> &strip, const mesh_point &to) {
        auto portals = std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>(strip.size());
        for (auto i = 0; i < strip.size() - 1; i++) {
            auto curr = strip[i], next = strip[i + 1];
            auto k = find_adjacent_triangle(triangles, curr, next);
            auto tr = coords[i];
            portals[i] = {tr[k], tr[mod3(k + 1)]};
        }
        auto end = interpolate_triangle(coords.back()[0], coords.back()[1], coords.back()[2], to.uv);
        portals.back() = {end, end};
        return portals;
    }

    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> unfold_funnel_portals(const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions,const std::vector<int> &strip, const mesh_point &start, const mesh_point &end) {
        auto coords = unfold_strip(triangles, positions, strip, start);
        return make_funnel_portals(triangles, coords, strip, end);
    }

    inline float intersect_segments(const Eigen::Vector2d &start1, const Eigen::Vector2d &end1, const Eigen::Vector2d &start2, const Eigen::Vector2d &end2) {
        if (end1 == start2) return 0;
        if (end2 == start1) return 1;
        if (start2 == start1) return 0;
        if (end2 == end1) return 1;
        auto a   = end1 - start1;    // direction of line a
        auto b   = start2 - end2;    // direction of line b, reversed
        auto d   = start2 - start1;  // right-hand side
        auto det = a[0] * b[1] - a[1] * b[0];
        return (a[0] * d[1] - a[1] * d[0]) / det;
    }

    static int max_curvature_point(const std::vector<funnel_point> &path) {
        // Among vertices around which the path curves, find the vertex
        // with maximum angle. We are going to fix that vertex. Actually, max_index is
        // the index of the first face containing that vertex.
        auto max_index = -1;
        auto max_angle = 0.0f;
        for (auto i = 1; i < path.size() - 1; ++i) {
            auto pos = path[i].pos;
            auto prev = path[i - 1].pos;
            auto next = path[i + 1].pos;
            auto v0 = (pos - prev).normalized();
            auto v1 = (next - pos).normalized();
            auto angle = 1 - v0.dot(v1);
            if (angle > max_angle) {
                max_index = path[i].face;
                max_angle = angle;
            }
        }
        return max_index;
    }

    static std::vector<float> funnel(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> &portals, int &max_index) {
        // Find straight path.
        Eigen::Vector2d start = {0, 0};
        auto apex_index = 0;
        auto left_index = 0;
        auto right_index = 0;
        auto apex = start;
        auto left_bound = portals[0].first;
        auto right_bound = portals[0].second;

        // Add start point.
        auto points = std::vector<funnel_point>{{apex_index, apex}};
        points.reserve(portals.size());

        // @Speed: is this slower than an inlined function?
        auto area = [](const Eigen::Vector2d a, const Eigen::Vector2d b, const Eigen::Vector2d c) {
            return cross(b - a, c - a);
        };

        for (auto i = 1; i < portals.size(); ++i) {
            auto left = portals[i].first, right = portals[i].second;
            // Update right vertex.
            if (area(apex, right_bound, right) <= 0) {
                if (apex == right_bound || area(apex, left_bound, right) > 0) {
                // Tighten the funnel.
                right_bound = right;
                right_index = i;
                } 
                else {
                    // Right over left, insert left to path and restart scan from
                    // portal left point.
                    if (left_bound != apex) {
                        points.push_back({left_index, left_bound});
                        // Make current left the new apex.
                        apex = left_bound;
                        apex_index = left_index;
                        // Reset portal
                        left_bound = apex;
                        right_bound = apex;
                        left_index = apex_index;
                        right_index = apex_index;
                        // Restart scan
                        i = apex_index;
                        continue;
                    }
                }
            }

            // Update left vertex.
            if (area(apex, left_bound, left) >= 0) {
                if (apex == left_bound || area(apex, right_bound, left) < 0) {
                    // Tighten the funnel.
                    left_bound = left;
                    left_index = i;
                } 
                else {
                    if (right_bound != apex) {
                        points.push_back({right_index, right_bound});
                        // Make current right the new apex.
                        apex = right_bound;
                        apex_index = right_index;
                        // Reset portal
                        left_bound = apex;
                        right_bound = apex;
                        left_index = apex_index;
                        right_index = apex_index;
                        // Restart scan
                        i = apex_index;
                        continue;
                    }
                }
            }
        }

        // This happens when we got an apex on the last edge of the strip
        if (points.back().pos != portals.back().first) {
            points.push_back({(int)portals.size() - 1, portals.back().first});
        }

        auto lerps = std::vector<float>();
        lerps.reserve(portals.size());
        for (auto i = 0; i < points.size() - 1; i++) {
            auto a = points[i].pos;
            auto b = points[i + 1].pos;
            for (auto k = points[i].face; k < points[i + 1].face; k++) {
                auto portal = portals[k];
                auto s = intersect_segments(a, b, portal.first, portal.second);
                lerps.push_back(std::clamp(s, 0.0f, 1.0f));
            }
        }

        auto index = 1;
        for (auto i = 1; i < portals.size(); ++i) {
            if ((portals[i].first == points[index].pos) || (portals[i].second == points[index].pos)) {
                points[index].face = i;
                index += 1;
            }
        }
        max_index = max_curvature_point(points);
        return lerps;
    }
    
    // Finds common edge between triangles
    Eigen::Vector2i common_edge(const Eigen::Vector3i &triangle0, const Eigen::Vector3i &triangle1) {
        for (auto i = 0; i < 3; i++) {
            for (auto k = 0; k < 3; k++) {
                if (triangle0[i] == triangle1[k] && triangle0[mod3(i + 1)] == triangle1[mod3(k + 2)])
                    return {triangle0[i], triangle0[mod3(i + 1)]};
            }
        }
        return {-1, -1};
    }

    // Triangle fan starting from a face and going towards the k-th neighbor face.
    std::vector<int> triangle_fan(const Eigen::MatrixXi &adjacencies, int face, int k, bool clockwise) {
        auto result = std::vector<int>{};
        result.push_back(face);
        auto prev = face;
        auto node = adjacencies.row(face)[k];
        auto offset = 2 - (int)clockwise;
        while (true) {
            if (node == -1) break;
            if (node == face) break;
            result.push_back(node);
            auto kk = find_in_vec(adjacencies.row(node), prev);
            kk   = mod3(kk + offset);
            prev = node;
            node = adjacencies.row(node)[kk];
        }
        return result;
    }

    static std::vector<int> fix_strip(const Eigen::MatrixXi &adjacencies, const std::vector<int> &strip, int index, int k, bool left) {
        auto face = strip[index];
        if (!left) k = mod3(k + 2);

        // Create triangle fan that starts at face, walks backward along the strip for
        // a while, exits and then re-enters back.
        auto fan = triangle_fan(adjacencies, face, k, left);

        // strip in the array of faces and fan is a loop of faces which has partial
        // intersection with strip. We wan to remove the intersection from strip and
        // insert there the remaining part of fan, so that we have a new valid strip.
        auto first_strip_intersection = index;
        auto first_fan_intersection   = 0;
        for (auto i = 1; i < fan.size(); i++) {
            auto fan_index   = i;
            auto strip_index = std::max(index - i, 0);
            if (strip_index < 0) break;
            if (fan[fan_index] == strip[strip_index]) {
                first_strip_intersection = strip_index;
                first_fan_intersection   = fan_index;
            } else {
                break;
            }
        }
        auto second_strip_intersection = index;
        auto second_fan_intersection   = 0;
        for (auto i = 0; i < fan.size(); i++) {
            auto fan_index   = (int)fan.size() - 1 - i;
            auto strip_index = index + i + 1;
            if (strip_index >= (int)strip.size()) break;
            if (fan[fan_index] == strip[strip_index]) {
                second_strip_intersection = strip_index;
                second_fan_intersection   = fan_index;
            } 
            else
                break;
        }

        auto result = std::vector<int>{};
        result.reserve(strip.size() + 12);
        // Initial part of original strip, until intersection with fan.
        for (auto i = 0; i < first_strip_intersection; ++i)
            result.push_back(strip[i]);

        // Append out-flanking part of fan.
        result.insert(result.end(), fan.begin() + first_fan_intersection,
            fan.begin() + second_fan_intersection);

        // Append remaining part of strip after intersection with fan.
        for (auto i = second_strip_intersection; i < strip.size(); ++i)
            result.push_back(strip[i]);

        return result;
    }

    static void straighten_path(geodesic_path &path, const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions, const Eigen::MatrixXi &adjacencies) {
        auto index = -1, vertex = -1;
        auto init_portals = unfold_funnel_portals(triangles, positions, path.strip, path.start, path.end);
        path.lerps = funnel(init_portals, index);

        for (auto i = 0; i < path.strip.size() * 2 && index != -1; i++) {
            auto new_vertex = -1;
            auto face = path.strip[index];
            auto next = path.strip[index + 1];
            auto edge = common_edge(triangles.row(face), triangles.row(next));
            auto flank_left = false;
            
            if (path.lerps[index] == 0) {
                new_vertex = edge[0];
                flank_left = false;
            } 
            else if (path.lerps[index] == 1) {
                new_vertex = edge[1];
                flank_left = true;
            }
            if (new_vertex == vertex) 
                break;
            
            vertex = new_vertex;

            path.strip = fix_strip(adjacencies, path.strip, index,
                find_in_vec(triangles.row(face), vertex), flank_left);

            auto portals = unfold_funnel_portals(
                triangles, positions, path.strip, path.start, path.end);
            path.lerps = funnel(portals, index);
        }
    }

    geodesic_path shortest_path(const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions, const Eigen::MatrixXi &adjacencies, const mesh_point &start, const mesh_point &end, const std::vector<int> &strip) {
        auto path = geodesic_path{};
        path.start = start;
        path.end = end;
        path.strip = strip;
        straighten_path(path, triangles, positions, adjacencies);
        return path;
    }

    geodesic_path compute_geodesic_path(const bezier_mesh& mesh, const mesh_point& start, const mesh_point& end, int thread_id=0) {
        auto path = geodesic_path{};
        if (start.face == end.face) {
            path.start = start;
            path.end = end;
            path.strip = {start.face};
            return path;
        }
        auto strip = compute_strip_tlv(mesh, end.face, start.face);
        path = shortest_path(mesh.triangles, mesh.positions, mesh.adjacencies, start, end, strip);
        return path;
    }

    bezier_segment get_control_polygon(const Gui_Spline& spline, int curve_id) {
        auto polygon = bezier_segment{};
        for (int i = 0; i < 4; ++i)
            polygon[i] = spline.control_points[curve_id * 3 + i];
        return polygon;
    }

    Eigen::Vector2i get_edge(const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions, const Eigen::MatrixXi &adjacencies, int f0, int f1) {
        auto k = find_in_vec(adjacencies.row(f0), f1);
        if (k == -1) 
            return {-1, -1};
        auto tr = triangles.row(f0);
        return Eigen::Vector2i(tr[k], tr[mod3(k + 1)]);
    }

    Eigen::MatrixXd path_positions(const geodesic_path &path, const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions, const Eigen::MatrixXi &adjacencies) {
        Eigen::MatrixXd result(path.lerps.size() + 2, 3);
        result.row(0) = eval_position(triangles, positions, path.start);
        for (auto i = 0; i < path.lerps.size(); i++) {
            auto e = get_edge(triangles, positions, adjacencies, path.strip[i], path.strip[i + 1]);
            if (e == Eigen::Vector2i(-1, -1)) 
                continue;
            auto x = path.lerps[i];
            Eigen::Vector3d p = lerp(Eigen::Vector3d(positions.row(e[0])), Eigen::Vector3d(positions.row(e[1])), x);
            result.row(i + 1) = p;
        }
        result.row(result.rows()-1) = eval_position(triangles, positions, path.end);
        return result;
    }

    Eigen::MatrixXd path_positions(const bezier_mesh& mesh, const geodesic_path& path) {
        return path_positions(path, mesh.triangles, mesh.positions, mesh.adjacencies);
    }

    Eigen::MatrixXd path_positions(const bezier_mesh& mesh, const mesh_path& path) {
        Eigen::MatrixXd positions = {path.points.size(), 3};
            for (int i = 0; i < positions.rows(); i++)
                positions.row(i) = eval_position(mesh, path.points[i]);
        return positions;
    }

    float path_length(const bezier_mesh& mesh, const geodesic_path& path) {
        auto positions = path_positions(path, mesh.triangles, mesh.positions, mesh.adjacencies);

        auto result = 0.0f;
        for (int i = 1; i < positions.rows(); ++i)
            result += (positions.row(i) - positions.row(i - 1)).norm();

        return result;
    }


    float path_length(const Eigen::MatrixXd& positions){
        auto result = 0.0f;
        for (int i = 1; i < positions.rows(); ++i)
            result += (positions.row(i) - positions.row(i - 1)).norm();

        return result;
    }

    std::vector<float> path_parameters(const Eigen::MatrixXd &positions) {
        auto len = 0.0f;
        auto parameter_t = std::vector<float>(positions.rows());
        for (auto i = 0; i < positions.rows(); i++) {
            if (i) 
                len += (positions.row(i) - positions.row(i - 1)).norm();
            parameter_t[i] = len;
        }
        for (auto &t : parameter_t){
            t /= len;
        }
        return parameter_t;
    }

    mesh_point eval_path_point(const geodesic_path &path, const Eigen::MatrixXi &triangles, const Eigen::MatrixXd &positions, const Eigen::MatrixXi &adjacencies, float t) {
        // strip with 1 triangle are trivial, just average the uvs
        if (path.start.face == path.end.face)
            return mesh_point{path.start.face, lerp(path.start.uv, path.end.uv, t)};
        
        // util function
        auto rotate = [](const Eigen::Vector3d &v, int k) {
            if (mod3(k) == 0)
                return v;
            else if (mod3(k) == 1)
                return Eigen::Vector3d(v[2], v[0], v[1]);
            else
                return Eigen::Vector3d(v[1], v[2], v[0]);
        };
        
        auto parameter_t = path_parameters(path_positions(path, triangles, positions, adjacencies));
        // find the point in the middle
        auto i = 0;
        for (; i < parameter_t.size() - 1; i++)
            if (parameter_t[i + 1] >= t)
                break;
        auto t_low = parameter_t[i], t_high = parameter_t[i + 1];
        auto alpha = (t - t_low) / (t_high - t_low);

        // alpha == 0 -> t_low, alpha == 1 -> t_high
        auto face = path.strip[i];
        auto uv_low = Eigen::Vector2d(0, 0);
        if (i == 0) 
            uv_low = path.start.uv;
        else {
            auto uvw = lerp(Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), 1 - path.lerps[i - 1]);
            auto prev_face = path.strip[i - 1];
            int k = find_in_vec(adjacencies.row(face), prev_face);
            uvw = rotate(uvw, k + 0); // originally k+2, due to uv order
            uv_low = Eigen::Vector2d(uvw[0], uvw[1]);
        }
        auto uv_high = Eigen::Vector2d(0, 0);
        if (i == parameter_t.size() - 2)
            uv_high = path.end.uv;
        else {
            auto uvw = lerp(Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), path.lerps[i]);
            auto next_face = path.strip[i + 1];
            auto k = find_in_vec(adjacencies.row(face), next_face);
            uvw = rotate(uvw, k + 0); // originally k+2, due to uv order
            uv_high = Eigen::Vector2d(uvw[0], uvw[1]);
        }

        auto uv = lerp(uv_low, uv_high, alpha);
        auto point = mesh_point{face, uv};
        return point;
    }

    inline mesh_point geodesic_lerp(const bezier_mesh& mesh, const mesh_point& start, const mesh_point& end, float t) {
        if (start.face == end.face)
            return mesh_point{start.face, lerp(start.uv, end.uv, t)};   

        auto path  = compute_geodesic_path(mesh, start, end);
        auto point = eval_path_point(path, mesh.triangles, mesh.positions, mesh.adjacencies, t);
        return point;
    }

    std::pair<bezier_segment, bezier_segment> subdivide_bezier_polygon(const bezier_mesh& mesh, const bezier_segment& input, float t) {
        auto Q0 = geodesic_lerp(mesh, input[0], input[1], t);
        auto Q1 = geodesic_lerp(mesh, input[1], input[2], t);
        auto Q2 = geodesic_lerp(mesh, input[2], input[3], t);
        auto R0 = geodesic_lerp(mesh, Q0, Q1, t);
        auto R1 = geodesic_lerp(mesh, Q1, Q2, t);
        auto S  = geodesic_lerp(mesh, R0, R1, t);
        return {{input[0], Q0, R0, S}, {S, R1, Q2, input[3]}};
    }

    std::vector<mesh_point> bezier_uniform(const bezier_mesh& mesh, const bezier_segment& control_points, const bezier_params& params) {
        auto segments = std::vector<bezier_segment>{control_points};
        auto result = std::vector<bezier_segment>();
        auto count = 0;
        for (auto subdivision = 0; subdivision < params.subdivisions; subdivision++) {
            result.clear();
            result.reserve(segments.size() * 2);
            for (auto i = 0; i < segments.size(); i++) {
                auto [split0, split1] = subdivide_bezier_polygon(mesh, segments[i], 0.5);
                count += 6;
                result.push_back(split0);
                result.push_back(split1);
            }
            swap(segments, result);
        }
        return {(mesh_point*)segments.data(), (mesh_point*)segments.data() + segments.size() * 4};
    }

    Eigen::MatrixXd append(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b){
        Eigen::MatrixXd result(a.rows() + b.rows(), b.cols());
        result << a, b;
        return result;
    }

    Eigen::MatrixXd make_polyline_positions(const bezier_mesh& mesh, const std::vector<mesh_point>& points) {
        Eigen::MatrixXd result(0,3);
        for (int i = 0; i < points.size() - 1; i++) {
            auto a = points[i];
            auto b = points[i + 1];
            auto path = compute_geodesic_path(mesh, a, b);
            result = append(result, path_positions(mesh, path));
        }
        return result;
    }


    void update_curve_shape(Gui_Spline& spline, const int curve_id, const bezier_mesh& mesh, const bezier_params& params) {
        if (curve_id < 0 || curve_id >= spline.curves.size()) 
            return;

        auto& curve = spline.curves[curve_id];
        auto polygon = get_control_polygon(spline, curve_id);

        auto& t0 = curve.tangents[0];
        t0.path = compute_geodesic_path(mesh, polygon[0], polygon[1]);
        t0.positions = path_positions(mesh, t0.path);
        t0.length = path_length(t0.positions);

        auto& t1 = curve.tangents[1];
        t1.path = compute_geodesic_path(mesh, polygon[3], polygon[2]);
        t1.positions = path_positions(mesh, t1.path);
        t1.length = path_length(t1.positions);
        auto points = std::vector<mesh_point>{};

        if (params.algorithm == spline_algorithm::de_casteljau_uniform)
            points = bezier_uniform(mesh, polygon, params);
        
        // altri algoritmi da portare
        // else if (params.algorithm == spline_algorithm::de_casteljau_adaptive) {
        //     points = bezier_adaptive(mesh, polygon, params);
        //     printf("num points: %d\n", (int)points.size());
        // } else if (params.algorithm == spline_algorithm::subdivision_uniform) {
        //     points = spline_subdivision_uniform(mesh, polygon, params.subdivisions);
        //     printf("num points: %d\n", (int)points.size());
        // } else if (params.algorithm == spline_algorithm::subdivision_adaptive) {
        //     points = spline_subdivision_adaptive(mesh, polygon, params);
        //     printf("num points: %d\n", (int)points.size());
        // }
        
        curve.positions = make_polyline_positions(mesh, points);
        if (curve.positions.size() == 0)
            curve.positions = eval_position(mesh, points[0]);
        
        // allocate the matrices for bezier points
        Eigen::MatrixXd newPos = Eigen::MatrixXd(spline.positions.rows()+curve.positions.rows(), spline.positions.cols());
        Eigen::MatrixXi newEdges = Eigen::MatrixXi(spline.edges.rows()+curve.positions.rows() - 1, spline.edges.cols());
        Eigen::MatrixXd newColors = Eigen::MatrixXd(spline.colors.rows()+curve.positions.rows() - 1, spline.colors.cols());
        newPos << spline.positions, curve.positions;
        for (int i=0; i<newEdges.rows(); i++){
            newEdges.row(i) = Eigen::Vector2i(i, i+1);
            newColors.row(i) = Eigen::Vector3d(1, 0, 0);
        }
        spline.positions = newPos;
        spline.edges = newEdges;
        spline.colors = newColors;
        
        std::vector<Eigen::MatrixXd> geodesic_polygon_pieces;
        int total_rows = 0;
        std::vector<int> rel_rows;
        // get a geodesic line for each control point segment
        for(int i=0; i<polygon.size()-1; i++){
            geodesic_polygon_pieces.push_back(path_positions(mesh, compute_geodesic_path(mesh, polygon[i], polygon[i+1])));
            rel_rows.push_back(total_rows);
            total_rows += geodesic_polygon_pieces[i].rows();
        }
        
        // merge the three lines into a single one
        Eigen::MatrixXd geodesic_polygon = Eigen::MatrixXd(total_rows,3);
        for(int i=0; i<geodesic_polygon_pieces.size(); i++){
            for(int j=0; j<geodesic_polygon_pieces[i].rows(); j++)
                geodesic_polygon.row(rel_rows[i] + j) = geodesic_polygon_pieces[i].row(j);
        }

        // allocate the matrices for the new control polygon
        Eigen::MatrixXd newPosCP = Eigen::MatrixXd(spline.cp_positions.rows()+geodesic_polygon.rows(), spline.cp_positions.cols());
        Eigen::MatrixXi newEdgesCP = Eigen::MatrixXi(spline.cp_edges.rows()+geodesic_polygon.rows() - 1, spline.cp_edges.cols());
        Eigen::MatrixXd newColorsCP = Eigen::MatrixXd(spline.cp_colors.rows()+geodesic_polygon.rows() - 1, spline.cp_colors.cols());
        newPosCP << spline.cp_positions, geodesic_polygon;
        for (int i=0; i<newEdgesCP.rows(); i++){
            newEdgesCP.row(i) = Eigen::Vector2i(i, i+1);
            newColorsCP.row(i) = Eigen::Vector3d(0, 0, 1);
        }

        spline.cp_positions = newPosCP;
        spline.cp_edges = newEdgesCP;
        spline.cp_colors = newColorsCP;
    }

    Eigen::Vector2d tangent_path_direction(const bezier_mesh& mesh, const geodesic_path& path, bool start = true) {
        auto find = [](const Eigen::Vector3i& vec, int x) {
            for (int i = 0; i < vec.size(); i++){
                if (vec[i] == x) 
                    return i;
            }
            return -1;
        };

        auto direction = Eigen::Vector2d{};

        if (start) {
            auto start_tr = triangle_coordinates(mesh.triangles, mesh.positions, path.start);
            
            if (path.lerps.empty())
                direction = interpolate_triangle(start_tr[0], start_tr[1], start_tr[2], path.end.uv);
            else {
                auto x = path.lerps[0];
                auto k = find(mesh.adjacencies.row(path.strip[0]), path.strip[1]);
                direction = lerp(start_tr[k], start_tr[(k + 1) % 3], x);
            }
        } 
        else {
            auto end_tr = triangle_coordinates(mesh.triangles, mesh.positions, path.end);

            if (path.lerps.empty())
                direction = interpolate_triangle(end_tr[0], end_tr[1], end_tr[2], path.start.uv);
            else {
                auto x = path.lerps.back();
                auto k = find(mesh.adjacencies.row(path.strip.rbegin()[0]), path.strip.rbegin()[1]);
                direction = lerp(end_tr[k], end_tr[(k + 1) % 3], 1 - x);
            }
        }
        return direction.normalized();
    }

    Eigen::Vector2d barycentric_coordinates(const Eigen::Vector2d &point, const Eigen::Vector2d &a, const Eigen::Vector2d &b, const Eigen::Vector2d &c) {
        auto  v0 = b - a, v1 = c - a, v2 = point - a;
        float d00 = v0.dot(v0);
        float d01 = v0.dot(v1);
        float d11 = v1.dot(v1);
        float d20 = v2.dot(v0);
        float d21 = v2.dot(v1);
        float denom = d00 * d11 - d01 * d01;
        Eigen::Vector2d result = {d11 * d20 - d01 * d21, d00 * d21 - d01 * d20};
        return result / denom;
    };

    // given a direction expressed in tangent space of the face start,
    // continue the path as straight as possible.
    geodesic_path straightest_path(const Eigen::MatrixXi& triangles, const Eigen::MatrixXd& positions, const Eigen::MatrixXi&adjacencies, const mesh_point &start, const Eigen::Vector2d& direction, float path_length) {
        auto path  = geodesic_path{};
        path.start = start;
        path.strip.push_back(start.face);

        unfold_triangle coords = triangle_coordinates(triangles, positions, start);
        auto prev_face = -2, face = start.face;
        auto len = 0.0f;

        // https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
        auto intersect = [](const Eigen::Vector2d &direction, const Eigen::Vector2d &left, const Eigen::Vector2d &right) {
            Eigen::Vector2d v1 = -left;
            Eigen::Vector2d v2 = right - left;
            Eigen::Vector2d v3 = {-direction[1], direction[0]};
            float t0 = (cross(v2, v1) / v2.dot(v3));
            auto t1 = -left.dot(v3) / v2.dot(v3);
            return std::pair<float, float>{t0, t1};
        };

        for (int it = 0; it < 10000 && len < path_length; it++) {
            // Given the triangle, find which edge is intersected by the line.
            for (auto k = 0; k < 3; ++k) {
                auto neighbor = adjacencies.row(face)[k];
                if (neighbor == prev_face) 
                    continue;
                auto left = coords[k];
                auto right = coords[mod3(k + 1)];
                auto [t0, t1] = intersect(direction, left, right);
                if (t0 > 0 && t1 >= 0 && t1 <= 1) {
                    len = t0;
                    if (t0 < path_length) {
                        path.lerps.push_back(t1);
                        // Step to next face.
                        prev_face = face;
                        if (neighbor == -1) {
                            path_length = len;
                            break;
                        }
                        coords = unfold_face(triangles, positions, coords, face, neighbor);
                        face   = adjacencies.row(face)[k];
                        path.strip.push_back(face);
                    }
                    break;
                }
            }
        }

        Eigen::Vector2d p = direction * path_length;
        auto uv  = barycentric_coordinates(p, coords[0], coords[1], coords[2]);
        path.end = {face, uv};
        return path;
    }

    geodesic_path continue_path(const bezier_mesh& mesh, const geodesic_path& path, float length) {
        auto direction = tangent_path_direction(mesh, path);
        if (length < 0)
            direction = -direction;
        return straightest_path(mesh.triangles, mesh.positions, mesh.adjacencies, path.start, direction, std::abs(length));
    }

    dual_geodesic_solver make_dual_geodesic_solver(const Eigen::MatrixXi &_triangles, const Eigen::MatrixXd &_positions, const Eigen::MatrixXi &_adjacencies) {
        
        auto get_triangle_center = [](const Eigen::MatrixXi  &triangles, const Eigen::MatrixXd &positions, int face) -> Eigen::Vector3d {
            auto  tr = triangles.row(face);
            std::vector<Eigen::Vector3d> pos = {positions.row(tr[0]), positions.row(tr[1]), positions.row(tr[2])};
            auto l0 = (pos[0] - pos[1]).norm();
            auto p0 = (pos[0] + pos[1]) / 2;
            auto l1 = (pos[1] - pos[2]).norm();
            auto p1 = (pos[1] + pos[2]) / 2;
            auto l2 = (pos[2] - pos[0]).norm();
            auto p2 = (pos[2] + pos[0]) / 2;
            return (l0 * p0 + l1 * p1 + l2 * p2) / (l0 + l1 + l2);
        };

        auto solver = dual_geodesic_solver{};
        solver.num_original_faces = (int)_triangles.size();
        auto triangles = _triangles;
        auto positions = _positions;

        auto adjacencies = _adjacencies;//face_adjacencies(triangles);

        solver.graph.resize(triangles.rows());
        for(auto i = 0; i < solver.graph.size(); ++i) {
            for (auto k = 0; k < 3; ++k) {
                solver.graph[i][k].node = adjacencies.row(i)[k];
                if (adjacencies.row(i)[k] == -1) {
                    solver.graph[i][k].length = flt_max;
                } 
                else {
                    solver.graph[i][k].length = (get_triangle_center(triangles, positions, i) - get_triangle_center(triangles, positions, adjacencies.row(i)[k])).norm();
                }
            }
        }
        return solver;
    }
    
}