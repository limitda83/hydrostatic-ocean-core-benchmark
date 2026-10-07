/************************************************************************
 *  Module: config                                                      *
 *  Description: Parse the Fortran namelist that tools/toml2nml.py      *
 *               generates from config/*.toml, so the CUDA backend is   *
 *               driven by the same configuration tree as every other   *
 *               backend (RULES.md R3).                                *
 *  Pipeline: toml2nml.py -> namelist -> config -> cfd_exp_cuda         *
 ************************************************************************/
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

struct RunConfig {
    int nx = 100, ny = 100;
    double lx = 1.0e7, ly = 1.0e7;
    double g = 9.80616, f0 = 1.0e-4, h0 = 1000.0;
    std::string scheme_name = "theta";
    double theta = 0.5, theta_cor = 0.5;
    int n_picard = 2;
    std::string solver_kind = "pcg_jacobi";
    double rtol = 1.0e-12;
    int max_iter = 2000;
    std::string case_name = "igw";
    int mode_x = 2, mode_y = 2, n_modes = 8;
    double slope = 1.0, eta0 = 1.0;
    double dt = 0.0, t_final = 0.0;
    int n_steps = 0, n_repeat = 5, n_warmup = 1;
    std::string out_prefix = "output/cuda_run";
    /* --- 3D (spec v0.2) --- */
    int nz = 1;                       /* 1 selects the 2D barotropic model */
    double theta_v = 0.5;
    double n2 = 0.0, nu = 0.0, kappa = 0.0, rho0 = 1025.0;
    double tau_x = 0.0, tau_y = 0.0, bottom_drag = 0.0;
    int mode_z = 1, n_split = 0;
    double u0 = 1.0;
    std::string tridiag_kernel = "column";
    /* How often the device-resident PCG copies its residual to the host.
     * Every copy is a full synchronisation, which is what makes the GPU slow
     * on this solve (docs/21 section 3); checking every N iterations trades a
     * few wasted iterations for N-1 fewer stalls. */
    int pcg_check_every = 5;
    /* "device" keeps alpha/beta in device memory; "host" copies every
     * inner product back, which is what OpenACC's reduction clause does. */
    std::string pcg_sync = "device";
};

namespace nml {

inline std::string trim(std::string s) {
    const char* ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

/* Fortran writes doubles with a 'd' exponent, which strtod does not accept. */
inline double to_double(std::string s) {
    std::replace(s.begin(), s.end(), 'd', 'e');
    std::replace(s.begin(), s.end(), 'D', 'e');
    return std::strtod(s.c_str(), nullptr);
}

inline std::string unquote(std::string s) {
    if (s.size() >= 2 && (s.front() == '\'' || s.front() == '"')) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

inline std::map<std::string, std::string> read(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "FATAL: cannot open namelist %s\n", path.c_str());
        std::exit(1);
    }
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        const size_t bang = line.find('!');
        if (bang != std::string::npos) line = line.substr(0, bang);
        line = trim(line);
        if (line.empty() || line[0] == '&' || line[0] == '/') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return kv;
}

}  // namespace nml

inline RunConfig read_config(const std::string& path) {
    const auto kv = nml::read(path);
    RunConfig c;
    auto has = [&](const char* k) { return kv.count(k) > 0; };
    auto num = [&](const char* k, double d) {
        return has(k) ? nml::to_double(kv.at(k)) : d;
    };
    auto str = [&](const char* k, const std::string& d) {
        return has(k) ? nml::unquote(kv.at(k)) : d;
    };

    c.nx = static_cast<int>(num("nx", c.nx));
    c.ny = static_cast<int>(num("ny", c.ny));
    c.lx = num("lx", c.lx);
    c.ly = num("ly", c.ly);
    c.g = num("g", c.g);
    c.f0 = num("f0", c.f0);
    c.h0 = num("h0", c.h0);
    c.scheme_name = str("scheme_name", c.scheme_name);
    c.theta = num("theta", c.theta);
    c.theta_cor = num("theta_cor", c.theta_cor);
    c.n_picard = static_cast<int>(num("n_picard", c.n_picard));
    c.solver_kind = str("solver_kind", c.solver_kind);
    c.rtol = num("rtol", c.rtol);
    c.max_iter = static_cast<int>(num("max_iter", c.max_iter));
    c.case_name = str("case_name", c.case_name);
    c.mode_x = static_cast<int>(num("mode_x", c.mode_x));
    c.mode_y = static_cast<int>(num("mode_y", c.mode_y));
    c.n_modes = static_cast<int>(num("n_modes", c.n_modes));
    c.slope = num("slope", c.slope);
    c.eta0 = num("eta0", c.eta0);
    c.dt = num("dt", c.dt);
    c.t_final = num("t_final", c.t_final);
    c.n_steps = static_cast<int>(num("n_steps", c.n_steps));
    c.n_repeat = static_cast<int>(num("n_repeat", c.n_repeat));
    c.n_warmup = static_cast<int>(num("n_warmup", c.n_warmup));
    c.out_prefix = str("out_prefix", c.out_prefix);
    c.nz = static_cast<int>(num("nz", c.nz));
    c.theta_v = num("theta_v", c.theta_v);
    c.n2 = num("n2", c.n2);
    c.nu = num("nu", c.nu);
    c.kappa = num("kappa", c.kappa);
    c.rho0 = num("rho0", c.rho0);
    c.tau_x = num("tau_x", c.tau_x);
    c.tau_y = num("tau_y", c.tau_y);
    c.bottom_drag = num("bottom_drag", c.bottom_drag);
    c.mode_z = static_cast<int>(num("mode_z", c.mode_z));
    c.n_split = static_cast<int>(num("n_split", c.n_split));
    c.u0 = num("u0", c.u0);
    c.tridiag_kernel = str("tridiag_kernel", c.tridiag_kernel);
    c.pcg_check_every = static_cast<int>(num("pcg_check_every", c.pcg_check_every));
    c.pcg_sync = str("pcg_sync", c.pcg_sync);

    /* Validation mirrors libs/core/schemes.py::SchemeParams.validate. */
    if (c.scheme_name != "fb" && c.scheme_name != "theta"
        && c.scheme_name != "split_explicit") {
        std::fprintf(stderr, "FATAL: unknown scheme %s "
                             "(expected fb|theta|split_explicit)\n",
                     c.scheme_name.c_str());
        std::exit(1);
    }
    if (c.scheme_name == "theta" && !(c.theta > 0.0 && c.theta <= 1.0)) {
        std::fprintf(stderr, "FATAL: theta must lie in (0,1]; got %g\n", c.theta);
        std::exit(1);
    }
    if (c.n_picard < 1 || c.n_steps < 1 || c.dt <= 0.0) {
        std::fprintf(stderr, "FATAL: dt, n_steps and n_picard must come from the driver\n");
        std::exit(1);
    }
    return c;
}
