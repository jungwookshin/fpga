#ifndef RTI_TRACK_HPP
#define RTI_TRACK_HPP

#include <rti_vec.hpp>

namespace rti{
	
	
	typedef
	enum{
		 PHOTON = 0,
		 ELECTRON =1,
		 PROTON=2,
		 NEUTRON=3
	} particle_t;
    //PDG[PHOTON] =
	//PDG[PROTON] = 2212 //

	
	///< status of particle
	///< alive
	///< created
	///< killed
	typedef
	enum{
		 ALIVE  = 0,
		 KILLED ,
		 STEP
	} status_t;

	///< Particle properties at a position
	///< Physics will propose next vertex
	///< Geometry will propose next vertex
	///< Step limiter will propose next vertex
	///< Magnetic field will propose next vertex
	///< Vertex doesn't include particle type
	template<typename T>
	struct vertex_t
	 {
		 T          ke   ;  //< kinetic energy
		 vec3<T>    pos  ;  //< position
		 vec3<T>    dir  ;  //< direction

		 CUDA_HOST_DEVICE
		 vertex_t<T>&
		 operator=(const vertex_t<T>& rhs)
		 {
			 ke  = rhs.ke ;
			 pos = rhs.pos;
			 dir = rhs.dir;
			 return *this;
		 }

	} ;

	///< Particle's dynamic information during transportation
	///< id: track ID
	///< sid: secondary ID
	template<typename T>
	struct track_t {
		uint32_t       id   ; //< track_id
		particle_t     type ; //< particle type
		vertex_t<T>    vtx  ; //< vertex information
	} ;

	//Forward declaration
	template<typename T, typename R> class rect3dmc; 

	///< Geometry token
	///< This will update as a track propagates
	template<typename T, typename R>
	struct token_t{
		rect3dmc<T, R>*  rct ; //< geometry where this track is in 
		vec3<uint16_t>   ijk ; //< i,j,k index of box
	};
	
	/*
	template<typename T>
	struct{
		//status: alive, stopped_physics, step_limit,
		//physics id
		T len ;
		T de  ;
		uint8_t n_secondaries;
	} step_t;
	*/
	

	/*
	template<typename T>
	void process_hit
	( track_t<T>& curr,
	  geometry<T>& geometry_point)
	{
		//geo.get_material

		//invoke physics: dedx (csda), inelastic, elastic, step_limit
		//

		//update the track

		//update the scorer

		//update geometry_point (check

		//store secondaries: track, step
	}
	*/

	///< Update track: calculate next position
	///< a part of physics process class?

	/*
	template<typename T>
	void
	update_track
	( track_t<T>& t,
	  vec3<T>& next_p,
	  T dedx)
	{
		T dis = (next_p - pos).norm() ;
		T dE  = dis/dedx ;
		//call proceshit
	}
	*/


	//geometrical_split_track: next position
}

#endif
