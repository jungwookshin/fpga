// This is a simple code from MGH/dicom-interface to test portability with FPGA H/W

#include <rti_beamlet.hpp>
#include <rti_track.hpp>
#include <rti_rect3dmc.hpp>


//typedef double phase_space_type;
typedef float phsp_t;

int main(int argc, char** argv){

	const size_t n_particles = (argc <=1 )? 1 : std::stoi(argv[1]) ;

	///
	///< 1. Voexel geometry (phantom)
	/// size: 20 cm x 20 cm x 20 cm and at (0,0,0)
	/// Box: edge -10 to 10 with step size 2 for X,Y,Z
	const uint8_t n_edges = 11;
    phsp_t xe1[n_edges] = {-10, -8, -6, -4, -2, 0, 2, 4, 6,8,10};
  	rti::rect3dmc<phsp_t, phsp_t> phantom(xe1, n_edges,
										  xe1, n_edges,
										  xe1, n_edges);
	phantom.fill_data(1.0); //let's assume 1.0 means water

	///< 2. Define a primary generator
	///< 2.1 energy distribution (normal distribution).
	rti::norm_1d<float> energy({100},{0.8}); //100 MeV +- 0.8 MeV
	
	///< 2.2 fluence distribution (emittance distribution)
	std::array<phsp_t,6> spot_mean = {0, 0, 15.0,  0, 0, -1.0};  //from (0,0,+15) cm with direction (0,0,-1)
	std::array<phsp_t,6> spot_sigm = {0.1, 0.1, 0, 0.01, 0.05, 0}; //spatial distribution (0.1 0.1 0) and direction distribution (0.01, 0.05, 0)
	std::array<phsp_t,2> corr      = {0.0, 0.0};
	rti::phsp_6d<phsp_t> emittance(spot_mean, spot_sigm, corr);

	///< 2.3 a spot
	// a spot has two distributions: energy and fluence
	rti::beamlet<phsp_t> spot(&energy, &emittance);

	// if you want to add rotation of gantry, couch, etc
	
	///< 2.1 generate vertices 

	rti::vertex_t<phsp_t>* primaries = new rti::vertex_t<phsp_t>[n_particles];
	for(size_t i = 0 ; i < n_particles ; ++i){
		primaries[i] = spot();
	}

	/// 3. simulation loop (to be parallelized)
	/// this loop is most time consuming part
	for (size_t i = 0 ; i  < n_particles ; ++i){
		// get primary vertex from CPU
		auto v0 = primaries[i];

		// calculate mean free path by a physics
		// (Jose may contribute here)
		// As we don't have physics implementation yet, 
		// let's just set 9 % of energy (MeV) for mean_free_path
		phsp_t mean_free_path = 0.09 * v0.ke ;  //cm

		// calculate next vertex based on mean_free_path
		rti::vertex_t<phsp_t> v1 = v0;
		v0.dir * mean_free_path ;
		v1.pos += (v0.dir * mean_free_path);
		
		// check next vertex is in phantom
		rti::vec3<uint16_t> ijk;
		bool is_intersect = phantom.determine_intersect_bounding_box(v0.pos, v1.pos, ijk);
		if (is_intersect){
			// print entering pixel id
			std::cout<<i<<"-th vertex enters phantom at ("
					 << ijk.x <<", "
				     << ijk.y <<", "
				     << ijk.z <<") index\n";
			
			// let's transport v0 through a phantom
			// to be updated.
		}
				
	}
	return 0;
}
