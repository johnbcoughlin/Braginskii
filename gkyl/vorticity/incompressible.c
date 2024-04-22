// This simulation case replicates the campaign titled KH_5

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_moment.h>
#include <gkyl_util.h>
#include <gkyl_wv_ten_moment.h>
#include <gkyl_wv_euler.h>

#include <gkyl_null_comm.h>

#ifdef GKYL_HAVE_MPI
#include <mpi.h>
#include <gkyl_mpi_comm.h>
#endif

#include <rt_arg_parse.h>

struct sf_ctx
{
  // Mathematical constants (dimensionless).
  double pi;
  
  // Physical constants (using normalized code units).
  double epsilon0; // Permittivity of free space.
  double mu0; // Permeability of free space.
  double mass_ion; // Ion mass.
  double charge_ion; // Ion charge.
  double mass_elc; // Electron mass.
  double charge_elc; // Electron charge.
  double oct;
  double opt;

  double n_ref; // Reference temperature
  double T_ref; // Reference temperature
  double B0; // Reference magnetic field strength.
  double beta; // Plasma beta.
  double gamma;
  double u_s;
  double u_V;
  double kx;
  double zeta;
  double eta;
  double alpha;
  double w;
  
  // Simulation parameters.
  int Nx; // Cell count (x-direction).
  int Ny; // Cell count (y-direction).
  double Lx; // Domain size (x-direction).
  double Ly; // Domain size (y-direction).
  double k0; // Closure parameter.
  double cfl_frac; // CFL coefficient.
  double t_end; // Final simulation time.
  int num_frames; // Number of output frames
  double vti;
  double vte;
  int n_moments;
};

struct sf_ctx
create_ctx(int sim_id)
{
  // Mathematical constants (dimensionless).
  double pi = M_PI;

  double oct_array[] = {0.5, 1.0, 2.0};
  double oct = oct_array[sim_id / 12];

  double zeta_array[] = {0.5, 1.0, 1.5};
  double zeta = zeta_array[(sim_id / 4) % 3];

  double u_s_factor_array[] = {0.2, -0.2};
  double u_s_factor = u_s_factor_array[(sim_id / 2) % 2];

  int n_moments_array[] = {5, 10};
  int n_moments = n_moments_array[sim_id % 2];

  double q = oct;

  // Physical constants (using normalized code units).
  double epsilon0 = 1.0; // Permittivity of free space.
  double mu0 = 1.0; // Permeability of free space.
  double mass_ion = 1.0; // Ion mass.
  double charge_ion = q; // Ion charge.
  double mass_elc = 1.0 / 1836.0; // Electron mass.
  double charge_elc = -q; // Electron charge.
  double B0 = 1.0; // Reference magnetic field strength.
  
  double T_ref = 1e-3;
  double n_ref = 1.0;

  // Simulation parameters.
  int Nx = 256; // Cell count (x-direction).
  int Ny = 256; // Cell count (y-direction).
  double Lx = 1.0; // Domain size (x-direction).
  double Ly = 1.2; // Domain size (y-direction).
  double cfl_frac = 1.0; // CFL coefficient.
  double t_end = 150.0; // Final simulation time.
  int num_frames = 50; // Number of output frames.

  double kx = 2*M_PI;
  double vti = sqrt(T_ref / mass_ion);
  double vte = sqrt(T_ref / mass_elc);
  double u_s = u_s_factor*vti; // Shear velocity
  double u_V = 0.1*vti; // Vorticity velocity
  double gamma = 0.25;
  double alpha = 0.04;
  double w = 2*alpha;

  struct sf_ctx ctx = {
    .pi = pi,
    .epsilon0 = epsilon0,
    .mu0 = mu0,
    .mass_ion = mass_ion,
    .charge_ion = charge_ion,
    .mass_elc = mass_elc,
    .charge_elc = charge_elc,
    .oct = charge_ion,
    .opt = charge_ion,
    .n_ref = n_ref,
    .T_ref = T_ref,
    .B0 = B0,
    .Nx = Nx,
    .Ny = Ny,
    .Lx = Lx,
    .Ly = Ly,
    .kx = kx,
    .u_s = u_s,
    .u_V = u_V,
    .zeta = zeta,
    .alpha = alpha,
    .w = w,
    .gamma = gamma,
    .vti = vti,
    .vte = vte,
    .cfl_frac = cfl_frac,
    .t_end = t_end,
    .num_frames = num_frames,
    .n_moments = n_moments,
  };

  return ctx;
}

void sf_ctx_release(struct sf_ctx ctx) {
}

double phat(double y, struct sf_ctx *app) {
    return 1.0 + app->gamma * tanh(y / app->alpha);
}
double ni0(double y, struct sf_ctx* app) {
    return app->n_ref * pow(phat(y, app), app->zeta);
}
double Ti0(double y, struct sf_ctx* app) {
    return app->T_ref * pow(phat(y, app), 1.0 - app->zeta);
}
double pi0(double y, struct sf_ctx* app) {
    return ni0(y, app) * Ti0(y, app);
}

double phi_Y(double y, struct sf_ctx* app) {
    return 1.0 + app->u_s * app->alpha / 2.0 * log(cosh(y / app->alpha));
}
double phi_X(double x, double y, struct sf_ctx* app) {
    double w = app->w;
    return 1.0 + app->u_V / app->kx * sin(app->kx * x) * exp(-y*y / (2*w*w));
}
double phistar(double x, double y, struct sf_ctx* app) {
    return phi_Y(y, app) * phi_X(x, y, app);
}

double Ex0(double x, double y, struct sf_ctx* app) {
    double h =  1e-6;
    return -(phistar(x+h, y, app) - phistar(x-h, y, app)) / (2.0 * h);
}
double Ey0(double x, double y, struct sf_ctx* app) {
    double h = 1e-6;
    return -(phistar(x, y+h, app) - phistar(x, y-h, app)) / (2.0 * h);
}
double rhoc(double x, double y, struct sf_ctx* app) {
    double h = 1e-4;
    double L = phistar(x+h, y, app) + phistar(x-h, y, app) + phistar(x, y+h, app) + phistar(x, y-h, app) - 4.0*phistar(x, y, app);
    return -L / (h*h) / app->opt;
}

double dpi_dz(double y, struct sf_ctx* app) {
    double h = 1e-6;
    return app->n_ref * app->T_ref * (phat(y+h, app) - phat(y-h, app)) / (2.0*h);
}
double uEx(double x, double y, struct sf_ctx* app) {
    return app->opt / app->oct * Ey0(x, y, app) / app->B0;
}
double uEy(double x, double y, struct sf_ctx* app) {
    return -app->opt / app->oct * Ex0(x, y, app) / app->B0;
}
double uidx(double y, struct sf_ctx *app) {
    double p_ref = app->n_ref * app->T_ref;
    return -p_ref / app->oct * dpi_dz(y, app) / app->B0;
}

double ne0(double x, double y, struct sf_ctx *app) {
    return (-app->charge_ion * ni0(y, app) + rhoc(x, y, app)) / app->charge_elc;
}
double Te0(double x, double y, struct sf_ctx *app) {
    return pi0(y, app) / ne0(x, y, app);
}

void
evalElcInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  double x = xn[0], y = xn[1];
  struct sf_ctx *app = ctx;

  double mass_elc = app -> mass_elc;

  double n = ne0(x, y, app);
  double ni = ni0(y, app);
  double pre = pi0(y, app);
  double rhoe = n * mass_elc; // Electron mass density.
  double uex = uEx(x, y, app) - (uidx(y, app) * ni / n);
  double uey = uEy(x, y, app);

  // Set electron mass density.
  fout[0] = rhoe;
  // Set electron momentum density.
  fout[1] = rhoe * uex; fout[2] = rhoe * uey; fout[3] = 0.0;
  if (app->n_moments == 10) {
      // Set electron pressure tensor.
      fout[4] = pre + rhoe * uex * uex; fout[5] = rhoe * uex * uey; fout[6] = 0.0;
      fout[7] = pre + rhoe * uey * uey; fout[8] = 0.0; 
      fout[9] = pre;
  } else if (app->n_moments == 5) {
    fout[4] = 1.5*pre + 0.5 * rhoe * (uex * uex + uey * uey);
  } else {
      exit(1);
  }
}

void
evalIonInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  double x = xn[0], y = xn[1];
  struct sf_ctx *app = ctx;

  double mass_ion = app -> mass_ion;

  double n = ni0(y, app);
  double T = Ti0(y, app);
  double pri = n * T; // Ion pressure (scalar).
  double rhoi = n * mass_ion; // Ion mass density
  double uix = uEx(x, y, app) + uidx(y, app);
  double uiy = uEy(x, y, app);

  // Set ion mass density.
  fout[0] = rhoi;
  // Set ion momentum density.
  fout[1] = rhoi * uix; fout[2] = rhoi * uiy; fout[3] = 0.0;
  if (app->n_moments == 10) {
      // Set ion pressure tensor.
      fout[4] = pri + rhoi * uix * uix; fout[5] = rhoi * uix * uiy; fout[6] = 0.0;
      fout[7] = pri + rhoi * uiy * uiy; fout[8] = 0.0; 
      fout[9] = pri;
  } else if (app->n_moments == 5) {
    fout[4] = 1.5*pri + 0.5 * rhoi * (uix * uix + uiy * uiy);
  } else {
      exit(1);
  }
}

void
evalFieldInit(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  double x = xn[0], y = xn[1];
  struct sf_ctx *app = ctx;

  double pi = app -> pi;

  double B0 = app -> B0;

  double Lx = app -> Lx;
  double Ly = app -> Ly;

  double Ex = Ex0(x, y, app);
  double Ey = Ey0(x, y, app);
  double Ez = 0.0;

  double Bx = 0.0;
  double By = 0.0;
  double Bz = B0;

  // Set electric field.
  fout[0] = Ex, fout[1] = Ey; fout[2] = Ez;
  // Set magnetic field.
  fout[3] = Bx, fout[4] = By; fout[5] = Bz;
  // Set correction potentials.
  fout[6] = 0.0; fout[7] = 0.0;
}

void
write_data(struct gkyl_tm_trigger* iot, gkyl_moment_app* app, double t_curr)
{
  if (gkyl_tm_trigger_check_and_bump(iot, t_curr)) {
    gkyl_moment_app_write(app, t_curr, iot -> curr - 1);
  }
}

int
main(int argc, char **argv)
{
    if (argc < 2) {
  printf("Expected sim_id as first argument\n");
        exit(1);
    }
  int sim_id = atoi(argv[1]);
  printf("sim_id: %d\n", sim_id);
  struct gkyl_app_args app_args = parse_app_args(argc, argv);
  app_args.use_mpi = 0;

#ifdef GKYL_HAVE_MPI
  printf("Have MPI\n", sim_id);
  if (app_args.use_mpi) {
    printf("Using MPI\n", sim_id);
    MPI_Init(&argc, &argv);
  }
#endif

  if (app_args.trace_mem) {
    gkyl_cu_dev_mem_debug_set(true);
    gkyl_mem_debug_set(true);
  }

  struct sf_ctx ctx = create_ctx(sim_id); // Context for initialization functions.

  int NX = APP_ARGS_CHOOSE(app_args.xcells[0], ctx.Nx);
  int NY = APP_ARGS_CHOOSE(app_args.xcells[1], ctx.Ny);  

  // Electron/ion equations.
  struct gkyl_wv_eqn* elc_eqn;
  struct gkyl_wv_eqn* ion_eqn;
  if (ctx.n_moments == 10) {
      elc_eqn = gkyl_wv_ten_moment_new(ctx.k0);
      ion_eqn = gkyl_wv_ten_moment_new(ctx.k0);
  } else {
      elc_eqn = gkyl_wv_euler_new(5.0/3.0, false);
      ion_eqn = gkyl_wv_euler_new(5.0/3.0, false);
  }
  
  struct gkyl_moment_species elc = {
    .name = "elc",
    .charge = ctx.charge_elc, .mass = ctx.mass_elc,
    .equation = elc_eqn,
    .evolve = true,
    .init = evalElcInit,
    .ctx = &ctx,

    .bcy = { GKYL_SPECIES_REFLECT, GKYL_SPECIES_REFLECT },
  };

  struct gkyl_moment_species ion = {
    .name = "ion",
    .charge = ctx.charge_ion, .mass = ctx.mass_ion,
    .equation = ion_eqn,
    .evolve = true,
    .init = evalIonInit,
    .ctx = &ctx,

    .bcy = { GKYL_SPECIES_REFLECT, GKYL_SPECIES_REFLECT },    
  };

  // Field.
  struct gkyl_moment_field field = {
    .epsilon0 = ctx.epsilon0, .mu0 = ctx.mu0,
    .mag_error_speed_fact = 1.0,
    
    .evolve = true,
    .init = evalFieldInit,
    .ctx = &ctx,
    
    .bcy = { GKYL_FIELD_PEC_WALL, GKYL_FIELD_PEC_WALL },
  };

  int nrank = 1; // Number of processes in simulation.
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
  }
#endif

  // Create global range.
  int cells[] = { NX, NY };
  int dim = sizeof(cells) / sizeof(cells[0]);
  struct gkyl_range global_r;
  gkyl_create_global_range(dim, cells, &global_r);

  // Create decomposition.
  int cuts[dim];
#ifdef GKYL_HAVE_MPI
  for (int d = 0; d < dim; d++) {
    if (app_args.use_mpi) {
      cuts[d] = app_args.cuts[d];
    }
    else {
      cuts[d] = 1;
    }
  }
#else
  for (int d = 0; d < dim; d++) {
    cuts[d] = 1;
  }
#endif

  struct gkyl_rect_decomp *decomp = gkyl_rect_decomp_new_from_cuts(dim, cuts, &global_r);

  // Construct communicator for use in app.
  struct gkyl_comm *comm;
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    comm = gkyl_mpi_comm_new( &(struct gkyl_mpi_comm_inp) {
        .mpi_comm = MPI_COMM_WORLD,
        .decomp = decomp
      }
    );
  }
  else {
    comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
        .decomp = decomp,
        .use_gpu = app_args.use_gpu
      }
    );
  }
#else
  comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .decomp = decomp,
      .use_gpu = app_args.use_gpu
    }
  );
#endif

  int my_rank;
  gkyl_comm_get_rank(comm, &my_rank);
  int comm_size;
  gkyl_comm_get_size(comm, &comm_size);

  int ncuts = 1;
  for (int d = 0; d < dim; d++) {
    ncuts *= cuts[d];
  }

  if (ncuts != comm_size) {
    if (my_rank == 0) {
      fprintf(stderr, "*** Number of ranks, %d, does not match total cuts, %d!\n", comm_size, ncuts);
    }
    goto mpifinalize;
  }

  // Moment app.
  struct gkyl_moment app_inp = {
    .name = "5m_wavy",

    .ndim = 2,
    .lower = { 0.0 * ctx.Lx, -0.5 * ctx.Ly},
    .upper = { 1.0 * ctx.Lx, 0.5 * ctx.Ly},
    .cells = { NX, NY },

    .num_periodic_dir = 1,
    .periodic_dirs = { 0 },
    .cfl_frac = ctx.cfl_frac,

    .num_species = 2,
    .species = { elc, ion },

    .field = field,

    .has_low_inp = true,
    .low_inp = {
      .local_range = decomp -> ranges[my_rank],
      .comm = comm
    }
  };

  // Create app object.
  gkyl_moment_app *app = gkyl_moment_app_new(&app_inp);

  // Initial and final simulation times.
  double t_curr = 0.0, t_end = ctx.t_end;

  // Create trigger for IO.
  int num_frames = ctx.num_frames;
  struct gkyl_tm_trigger io_trig = { .dt = t_end / num_frames };

  // Initialize simulation.
  gkyl_moment_app_apply_ic(app, t_curr);
  write_data(&io_trig, app, t_curr);

  // Compute estimate of maximum stable time-step.
  double dt = gkyl_moment_app_max_dt(app);

  long step = 1;
  while ((t_curr < t_end) && (step <= app_args.num_steps)) {
    gkyl_moment_app_cout(app, stdout, "Taking time-step %ld at t = %g ...", step, t_curr);
    struct gkyl_update_status status = gkyl_moment_update(app, dt);
    gkyl_moment_app_cout(app, stdout, " dt = %g, but we said dt = %g\n", status.dt_actual, dt);
    
    if (!status.success) {
      gkyl_moment_app_cout(app, stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }

    t_curr += status.dt_actual;
    dt = status.dt_suggested / 3.0;

    write_data(&io_trig, app, t_curr);

    step += 1;
  }

  write_data(&io_trig, app, t_curr);
  gkyl_moment_app_stat_write(app);

  struct gkyl_moment_stat stat = gkyl_moment_app_stat(app);

  gkyl_moment_app_cout(app, stdout, "\n");
  gkyl_moment_app_cout(app, stdout, "Number of update calls %ld\n", stat.nup);
  gkyl_moment_app_cout(app, stdout, "Number of failed time-steps %ld\n", stat.nfail);
  gkyl_moment_app_cout(app, stdout, "Species updates took %g secs\n", stat.species_tm);
  gkyl_moment_app_cout(app, stdout, "Field updates took %g secs\n", stat.field_tm);
  gkyl_moment_app_cout(app, stdout, "Source updates took %g secs\n", stat.sources_tm);
  gkyl_moment_app_cout(app, stdout, "Total updates took %g secs\n", stat.total_tm);

  // Free resources after simulation completion.
  gkyl_wv_eqn_release(elc_eqn);
  gkyl_wv_eqn_release(ion_eqn);
  gkyl_rect_decomp_release(decomp);
  gkyl_comm_release(comm);
  gkyl_moment_app_release(app);  
  sf_ctx_release(ctx);
  
mpifinalize:
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Finalize();
  }
#endif
  
  return 0;
}
