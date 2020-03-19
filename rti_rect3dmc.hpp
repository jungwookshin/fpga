#ifndef RTI_RECT3DMC_H
#define RTI_RECT3DMC_H

/// \file
///
/// rectlinear geometry to represent CT, doe, DVF, etc.

#include <cmath>
#include <array>
#include <valarray>
#include <sstream>
#include <fstream>
#include <map>
#include <algorithm>

#include <rti_common.hpp>
#include <rti_vec.hpp>

namespace rti{

/// \class rect3dmc
/// \tparam T for grid values, e.g., dose, HU, vector
/// \tparam R for grid coordinates, float, double, etc.
template<typename T, typename R>
class rect3dmc {

public:

	///< enumerate to indicate cell-side
	///< XP, XM : a YZ plane at (XP)lus and (XM)inus
	///< YP, YM : a YZ plane at (YP)lus and (YM)inus
	///< ZP, ZM : a YZ plane at (ZP)lus and (ZM)inus 
	typedef enum
	{
		 XP=0,
		 YP  ,
		 ZP  ,
		 XM  ,    
		 YM  ,
		 ZM
	} cell_side;

	
	/// Data of eight corners around a center
	/// These data points are used for interpolation
	/// 0: lower, 1: upper
	/// X: 1, 0, 0 Y: 0, 1, 0 Z: 0, 0, 1 => c111
  	typedef struct{
		T c000;
		T c100;
		T c110;
		T c010;
		T c001;
		T c101;
		T c111;
		T c011;
	} cell_data;

	/// Position of neighbor pixels
	/// 0: lower, 1: upper
	/// X: 1,0 Y: 1,0 Z: 1,0
	typedef struct{
		R x0;
		R x1;
		R y0;
		R y1;
		R z0;
		R z1;
	} cell_center ;

	const vec3<R> nx_=vec3<R>(1,0,0);
	const vec3<R> ny_=vec3<R>(0,1,0);
	const vec3<R> nz_=vec3<R>(0,0,1);
	
	//edges
	//plan  z
	//plan: z0 -> c000 -> c010 -> c110 -> c100
	//plan: z1 -> 
protected:

	///< Number of data points (pixels)
	/// dim_.x, dim_.y, dim_.z
	rti::vec3<uint16_t> dim_; 

	/// positions of data points ~ pixel center
	/// calculated from xe_, ye_, ze_,
	/// acsending order
	R* x_ = nullptr;  ///< x axis values.
	R* y_ = nullptr;  ///< y axis values.
	R* z_ = nullptr;  ///< z axis values.
  
	/// the edge positions of pixels
	/// Bounding box xe_[0], xe_[max]
	/// number of edge points is dim_.x/y/z + 1
	/// acsending order
	R* xe_ = nullptr; ///< x_min, ..., x_max
	R* ye_ = nullptr; ///< y_min, ..., y_max
	R* ze_ = nullptr; ///< z_min, ..., z_min

	rti::vec3<R> B000_ ; ///< coner x_min, y_min, z_min
	rti::vec3<R> B111_ ; ///< coner x_max, y_max, z_max	
	
	///< A flag when input data points are reversed.
	//or set one of following flag
	bool flip_[3] = {false, false, false};

    ///< Data in this rectlinear, dim_.x*dim_.y*dim_.z
	T* data_ = nullptr;

	///< Calculate data points (x_/y_/z_)
	CUDA_HOST_DEVICE
	void
	calculate_pixel_center(void)
	{
		x_ = new R[dim_.x];
		y_ = new R[dim_.y];
		z_ = new R[dim_.z];

		for(uint16_t i=0 ; i < dim_.x ; ++i) x_[i] = 0.5*(xe_[i+1] + xe_[i]);
		for(uint16_t i=0 ; i < dim_.y ; ++i) y_[i] = 0.5*(ye_[i+1] + ye_[i]);
		for(uint16_t i=0 ; i < dim_.z ; ++i) z_[i] = 0.5*(ze_[i+1] + ze_[i]);
	}

	///< Calculate C000/C111
	CUDA_HOST_DEVICE
	void
	calculate_bounding_box(void)
	{
		B000_.x = xe_[0];
		B000_.y = ye_[0];
		B000_.z = ze_[0];
		B111_.x = xe_[dim_.x];
		B111_.y = ye_[dim_.y];
		B111_.z = ze_[dim_.z];
	}

	///< delete data
	CUDA_HOST_DEVICE
	void
	delete_data_if_used(void)
	{
		if( data_ != nullptr ) delete[] data_;      
	}

public:

	///< Default constructor only for child classes
    ///cuda_host_device or cuda_host
	/// note (Feb27,2020): it may be uesless
	CUDA_HOST_DEVICE
    rect3dmc()
    {;}

    /// Construct a rectlinear grid from vectors of x/y/z
    /// \param x,y,z  1D vector of EDGE points of voxels along x-axis
    /// for example, -1,0,1 mean TWO voxels along x.
    CUDA_HOST
    rect3dmc
	(std::vector<R>& xe,
	 std::vector<R>& ye,
	 std::vector<R>& ze)
    {
        xe_ = new R[xe.size()];
        ye_ = new R[ye.size()];
        ze_ = new R[ze.size()];

        for(uint16_t i=0 ; i < xe.size() ; ++i) xe_[i] = xe[i];
        for(uint16_t i=0 ; i < ye.size() ; ++i) ye_[i] = ye[i];
        for(uint16_t i=0 ; i < ze.size() ; ++i) ze_[i] = ze[i];

        dim_.x = xe.size()-1;
        dim_.y = ye.size()-1;
        dim_.z = ze.size()-1;

		this->calculate_pixel_center();
		this->calculate_bounding_box();
	}

    /// Construct a rectlinear grid from array of x/y/z with their size
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    CUDA_HOST_DEVICE
    rect3dmc
	(const R xe[], const uint16_t n_xe,
	 const R ye[], const uint16_t n_ye,
	 const R ze[], const uint16_t n_ze)
    {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        for(uint16_t i=0 ; i < n_xe ; ++i) xe_[i] = xe[i];
        for(uint16_t i=0 ; i < n_ye ; ++i) ye_[i] = ye[i];
        for(uint16_t i=0 ; i < n_ze ; ++i) ze_[i] = ze[i];

        dim_.x = n_xe-1;
        dim_.y = n_ye-1;
        dim_.z = n_ze-1;

		this->calculate_pixel_center();
		this->calculate_bounding_box();
    }

    ///< Copy constructor
    CUDA_HOST_DEVICE
    rect3dmc(rect3dmc& c)
    {
        dim_ = c.dim_;

        xe_ = new R[dim_.x+1];
        ye_ = new R[dim_.y+1];
        ze_ = new R[dim_.z+1];

        x_ = new R[dim_.x];
        y_ = new R[dim_.y];
        z_ = new R[dim_.z];

        for(uint16_t i=0 ; i < dim_.x ; ++i) x_[i] = c.x_[i];
        for(uint16_t i=0 ; i < dim_.y ; ++i) y_[i] = c.y_[i];
        for(uint16_t i=0 ; i < dim_.z ; ++i) z_[i] = c.z_[i];

		for(uint16_t i=0 ; i < dim_.x+1 ; ++i) xe_[i] = c.xe_[i];
        for(uint16_t i=0 ; i < dim_.y+1 ; ++i) ye_[i] = c.ye_[i];
        for(uint16_t i=0 ; i < dim_.z+1 ; ++i) ze_[i] = c.ze_[i];

		this->calculate_bounding_box();
    }

    ///< Destructor releases dynamic allocation for x/y/z coordinates
    CUDA_HOST_DEVICE
    ~rect3dmc()
	{
        delete[] x_;
        delete[] y_;
        delete[] z_;
		delete[] xe_;
		delete[] ye_;
		delete[] ze_;
		this->delete_data_if_used();
    }

   
	///< Return the interpolated value for given point
    /// \param x, y, z are for position
	CUDA_HOST_DEVICE
    virtual T
    operator()
	(const R x,
	 const R y,
	 const R z)
    {
        return operator()(rti::vec3<R>(x,y,z));
    }

    /// Returns the interpolated value for given point, x, y, z
    /// \param pos is a type of rti::vec3<R>.
	CUDA_HOST_DEVICE
    virtual T
    operator()
    (const rti::vec3<R>& pos)
    {
        rti::vec3<uint16_t> c000_idx = this->find_c000_index(pos);

		const cell_center cell_pts = this->get_cell_center(c000_idx);
		const cell_data   corner   = this->get_cell_data(c000_idx); 
	
        R xd = (pos.x- cell_pts.x0)/( cell_pts.x1 - cell_pts.x0 ) ;
        R yd = (pos.y- cell_pts.y0)/( cell_pts.y1 - cell_pts.y0 ) ;
        R zd = (pos.z- cell_pts.z0)/( cell_pts.z1 - cell_pts.z0 ) ;

        T c00 = corner.c000*(1.0-xd) + corner.c100*xd;
        T c10 = corner.c010*(1.0-xd) + corner.c110*xd;

        T c01 = corner.c001*(1.0-xd) + corner.c101*xd;
        T c11 = corner.c011*(1.0-xd) + corner.c111*xd;

        T c0 = c00*(1.0-yd) + c10*yd;
        T c1 = c01*(1.0-yd) + c11*yd;

        return c0*(1.0-zd)  + c1*zd;

    }

    /// Returns the data value for given x/y/z index
    /// \param[in] p index, p[0], p[1], p[2] for x, y, z.
	CUDA_HOST_DEVICE
    virtual	const T
    operator[]
    (const rti::vec3<uint16_t> p)
    {
		return data_[ijk2cnb(p.x, p.y, p.z)];
    }

	/// Returns data
    /// \return pointer of data
	CUDA_HOST_DEVICE
    const T*
	get_data() const
    {
        return data_;
    }

    /// Returns x center positions
    /// \return x_
	CUDA_HOST_DEVICE
    const R*
	get_x() const
    {
        return x_;
    }

    /// Returns y center positions
    /// \return y_
	CUDA_HOST_DEVICE
    const R*
	get_y() const
    {
        return y_;
    }

    /// Returns z center positions
    /// \return z_
	CUDA_HOST_DEVICE
    const R*
	get_z() const
    {
        return z_;
    }

    /// Returns x center positions
    /// \return xe_
	CUDA_HOST_DEVICE
    const R*
	get_x_edges() const
    {
        return xe_;
    }

    /// Returns y center positions
    /// \return ye_
	CUDA_HOST_DEVICE
    const R*
	get_y_edges() const
    {
        return ye_;
    }

    /// Returns z center positions
    /// \return ze_
	CUDA_HOST_DEVICE
    const R*
	get_z_edges() const
    {
        return ze_;
    }

    /// Returns a total of 8 coner data of a cell
    /// \param c000_idx an array of cell ids in x, y, and z.
    /// \return coner values of a cell, C000, C001, C100, etc.
    //virtual std::array<T, 8>. we can't use std::array<T, 8> due to GPUXS
    CUDA_HOST_DEVICE
    virtual
	cell_data
    get_cell_data
    (const rti::vec3<uint16_t>& c000_idx)
    {
        uint16_t x0 = c000_idx.x;
        uint16_t x1 = x0 + 1;
        uint16_t y0 = c000_idx.y;
        uint16_t y1 = y0 + 1;
        uint16_t z0 = c000_idx.z;
		uint16_t z1 = z0 + 1;
		cell_data t;
		t.c000 = data_[ijk2cnb(x0,y0,z0)];
		t.c100 = data_[ijk2cnb(x1,y0,z0)];
		t.c110 = data_[ijk2cnb(x1,y1,z0)];
		t.c010 = data_[ijk2cnb(x0,y1,z0)];
		t.c001 = data_[ijk2cnb(x0,y0,z1)];
		t.c101 = data_[ijk2cnb(x1,y0,z1)];
		t.c111 = data_[ijk2cnb(x1,y1,z1)];
		t.c011 = data_[ijk2cnb(x0,y1,z1)];
		return t;
    }


    /// Returns a total of 6 values of position
    /// \param c000_idx an array of cell index (x, y, and z).
    /// \return x0 and x1, y0 and y1, and z0 and z1.
	CUDA_HOST_DEVICE
    virtual
    cell_center
    get_cell_center
    (const rti::vec3<uint16_t>& c000_idx) const
    {
		///if c000_idx is at boundary, xpi+1 returns 
        uint16_t xpi = c000_idx.x ;
        uint16_t ypi = c000_idx.y ;
        uint16_t zpi = c000_idx.z ;
		cell_center t;
		t.x0 = x_[xpi];
		t.x1 = x_[xpi+1];
		t.y0 = y_[ypi];
		t.y1 = y_[ypi+1];
		t.z0 = z_[zpi];
		t.z1 = z_[zpi+1];
    }

	///< Calculate intersection
	//CUDA_HOST_DEVICE
	//virtual
	//R
	//calculate_intersect
	
	///< Determine two points intersect
	///< \param v0 : starting point
	///< \param v1 : ending point
	///< \param ijk : this will be set if v1 - v0 intersects a plane
	//r = ( B000_ - v0).dot(nx_)/(dir.dot(nx_));
	//formular: dir.dot(n_x) = dir.x

	CUDA_HOST_DEVICE
	virtual
	bool
	determine_intersect_bounding_box
	(const rti::vec3<R>& v0,
	 const rti::vec3<R>& v1,
	 rti::vec3<uint16_t>& ijk)
	{
		/// Direction vector (don't normalize)
		rti::vec3<R> v0tov1 = (v1 - v0);

		/// t : if intersect, the value is recorded
		/// -1 by default meaning no-interect
		R t[6] = {-1,-1,-1,-1,-1,-1}   ;		
		
		//1. YZ plane at +x and -x:
		if (v0tov1.x != 0.){
			/// YZ plane at -X
			R r = ( B000_ - v0).dot(nx_)/v0tov1.x ;
			rti::vec3<R> p = v0 + v0tov1*r;
			if ( this->is_in_bbox_yz(p) && r < 1 ) t[XM] = r; //intersect
			
			/// YZ plane at +X
			r = ( B111_ - v0).dot(nx_)/v0tov1.x ;
			p = v0 + v0tov1*r;
			if ( this->is_in_bbox_yz(p) && r < 1 ) t[XP] = r; //intersect
		}

		//2. ZX plane at +y and -y:
		if (v0tov1.y != 0.){
			/// ZX plane at -Y
			R r = ( B000_ - v0).dot(ny_)/v0tov1.y ;
			rti::vec3<R> p = v0 + v0tov1*r;
			if ( this->is_in_bbox_zx(p) && r < 1 ) t[YM] = r;
				 
			/// ZX plane at +Y
			r = ( B111_ - v0).dot(ny_)/v0tov1.y ;
			p = v0 + v0tov1*r;
			if ( this->is_in_bbox_zx(p) && r < 1 ) t[YP] = r;
		}

		//3. XY plane at +z and -z:
		if (v0tov1.z != 0.){
			/// XY plane at -Z
			R r = ( B000_ - v0).dot(nz_)/v0tov1.z ;
			rti::vec3<R> p = v0 + v0tov1*r;
			if ( this->is_in_bbox_xy(p) && r < 1 ) t[ZM] = r;

			/// XY plane at +Z
			r = ( B111_ - v0).dot(nz_)/v0tov1.z ;
			p = v0 + v0tov1*r;
			if ( this->is_in_bbox_xy(p) && r < 1 ) t[ZP] = r;
		}

		/// Find smallest t for a plane with which v0-v1 intersects
		R min_t = 1 ;
		uint8_t which_side = 0;
		for(uint8_t i = 0 ; i <= ZP ; ++i){
			if (t[i] == -1) continue;
			if (t[i] < min_t){
				min_t = t[i];
				which_side  = i;
			} 
		}

		/// no-intersect 
		if (min_t == 1) return false;

		/// intersect
		// let's calculate i,j,k of pixel that ray intersect
		rti::vec3<R> ip = v0 + v0tov1*min_t;

		switch(which_side){
		case XM: //-X, 
			{
				ijk.x = 0;
				ijk.y = this->find_c000_y_index(ip.y); 
				ijk.z = this->find_c000_z_index(ip.z); 
			}
			break;
		case XP: //+X
			{
				ijk.x = dim_.x -1  ; //-1 is needed
				ijk.y = this->find_c000_y_index(ip.y); 
				ijk.z = this->find_c000_z_index(ip.z); 
			}
			break;
		case YM: //-Y, 
			{
				ijk.x = this->find_c000_x_index(ip.x); 
				ijk.y = 0 ;
				ijk.z = this->find_c000_z_index(ip.z); 
			}
			break;
		case YP: //+Y
			{
				ijk.x = this->find_c000_x_index(ip.x); 
				ijk.y = dim_.y - 1 ;
				ijk.z = this->find_c000_z_index(ip.z); 
			}
			break;
		case ZM: //-Z, 
			{
				ijk.x = this->find_c000_x_index(ip.x); 
				ijk.y = this->find_c000_y_index(ip.y); 
				ijk.z = 0 ;
			}
			break;
		case ZP: //+Z
			{
				ijk.x = this->find_c000_x_index(ip.x); 
				ijk.y = this->find_c000_y_index(ip.y); 
				ijk.z = dim_.z -1 ;
			}
			break;
		}

		/*
		std::cout<<"t[0]: " << t[0] << "\n";
		std::cout<<"t[1]: " << t[1] << "\n";
		std::cout<<"t[2]: " << t[2] << "\n";
		std::cout<<"t[3]: " << t[3] << "\n";
		std::cout<<"t[4]: " << t[4] << "\n";
		std::cout<<"t[5]: " << t[5] << "\n";
		std::cout<<"min_t: " << min_t << "\n";
		std::cout<<"intersect point: \n";
		ip.dump();
		std::cout<<"ijk: \n";
		ijk.dump();
		*/
		
		return true ; 
	}
    
    /// Returns a coner index (x,y,z) where a given position is placed within (x+1,y+1,z+1)
    /// \param pos cartesian coordinate of x, y, and z.
    /// \return x,y,z indices of a cell
	CUDA_HOST_DEVICE
    virtual
    rti::vec3<uint16_t>
    find_c000_index
    (const rti::vec3<R>& pos)
    {
        rti::vec3<uint16_t> c000_idx;
        c000_idx.x = this->find_c000_x_index(pos.x);
        c000_idx.y = this->find_c000_y_index(pos.y);
        c000_idx.z = this->find_c000_z_index(pos.z);

        return c000_idx;
    }

    /// Returns a coner index of x
    /// \param pos cartesian coordinate of x
    /// \return x index of a cell, 0 or dim_.x +1 for outside
	///In case this code runs on GPU, consider to implement binary_search algorithm or use thrust
	/// but thrust performance is not so good from
	/// https://groups.google.com/forum/#!topic/thrust-users/kTX6lgntOAc
	CUDA_HOST_DEVICE
    inline virtual
    int16_t
    find_c000_x_index
    (const R& x)
    {
		// Following is for CPU version
        //R* i = std::lower_bound(xe_, xe_+ dim_.x + 1, x, std::less_equal<R>());
		//return i - x_ - 1 ;

		//note: if x < xe_[0] => throw or error
		int16_t i = 0;
		do{
			if (x < xe_[i]) return i-1;
		} while ( i++ <= dim_.x );
		return i-1; //instead, throw?
    }

    /// Returns a coner index of y
    /// \param pos cartesian coordinate of y
    /// \return y index of a cell
	CUDA_HOST_DEVICE
    inline virtual
    int16_t
    find_c000_y_index
    (const R& y)
    {
        //R* j = std::lower_bound(y_, y_+dim_.y, y, std::less_equal<R>()); ///CPU version
        //return j - y_ -1 ;
		int16_t i = 0;
		do{
			if (y < ye_[i]) return i-1;
		} while ( i++ <= dim_.y );
		return i-1;
    }

    /// Returns a coner index of z
    /// \param pos cartesian coordinate of z
    /// \return z index of a cell
	CUDA_HOST_DEVICE
    inline virtual
    int16_t
    find_c000_z_index
    (const R& z)
    {
        //R* k = std::lower_bound(z_, z_+dim_.z, z, std::less_equal<R>());
        //return k - z_ -1 ;
		int16_t i = 0;
		do{
			if (z < ze_[i]) return i-1;
		} while ( i++ <= dim_.z );
		return i-1;
    }

    /// Returns whether the point is in the grid or not
    /// \param pos cartesian coordinate of x,y,z
    /// \return true if the p is in grid or false
    CUDA_HOST_DEVICE
    inline virtual
    bool
    is_in_point
    (const vec3<R>& p)
    {
        ///< check p is inside of pixel grid not bounding box
        ///< Min/Max of src
        if( p.x < x_[0] || p.x > x_[dim_.x-1]) return false;
        if( p.y < y_[0] || p.y > y_[dim_.y-1]) return false;
        if( p.z < z_[0] || p.z > z_[dim_.z-1]) return false;
        return true;
    }

    /// Returns whether the point is in the bounding box
    /// \param pos cartesian coordinate of x,y,z
    /// \return true if the p is in grid or false
    CUDA_HOST_DEVICE
    inline virtual
    bool
    is_in_bbox(const vec3<R>& p)
    {
        if( p.x < xe_[0] || p.x > xe_[dim_.x]) return false;
        if( p.y < ye_[0] || p.y > ye_[dim_.y]) return false;
        if( p.z < ze_[0] || p.z > ze_[dim_.z]) return false;
        return true;
    }

    /// Returns whether the point is in the bounding box of x
    /// \param pos cartesian coordinate of x
    /// \return true if the p is in Bounding Box X or false
    CUDA_HOST_DEVICE
    inline virtual
    bool
    is_in_bbox_yz(const rti::vec3<R>& p)
    {
		if( p.y < ye_[0] || p.y > ye_[dim_.y]) return false;
        if( p.z < ze_[0] || p.z > ze_[dim_.z]) return false;
        return true;
    }

	/// Returns whether the point is in the bounding box of y
    /// \param pos cartesian coordinate of x
    /// \return true if the p is in Bounding Box X or false
    CUDA_HOST_DEVICE
    inline virtual
    bool
    is_in_bbox_zx(const rti::vec3<R>& p)
    {
		if( p.z < ze_[0] || p.z > ze_[dim_.z]) return false;
		if( p.x < xe_[0] || p.x > xe_[dim_.x]) return false;
        return true;
    }

    /// Returns whether the point is in the bounding box of z
    /// \param pos cartesian coordinate of x
    /// \return true if the p is in Bounding Box X or false
    CUDA_HOST_DEVICE
    inline virtual
    bool
    is_in_bbox_xy(const rti::vec3<R>& p)
    {
        if( p.x < xe_[0] || p.x > xe_[dim_.x]) return false;
		if( p.y < ye_[0] || p.y > ye_[dim_.y]) return false;
        return true;
    }

    
    /// Returns the center position of a rect
    /// \note more accurately, the center should be middle of edge
    /// not middle of first and last point cause the thickness may be differents
    CUDA_HOST_DEVICE
    rti::vec3<R>
    get_center()
    {
        return rti::vec3<R>(
            0.5*(xe_[0]+xe_[dim_.x]),
            0.5*(ye_[0]+ye_[dim_.y]),
            0.5*(ze_[0]+ze_[dim_.z])
         );
    }

    /// Returns size of box
    /// note: distance between edges
    /// the firt/last pixel thickness is assumed to be
    /// a half of distance between first and second or last and before-last
    CUDA_HOST_DEVICE
    rti::vec3<R>
    get_size()
    {
        R Lx =  xe_[dim_.x] - xe_[0] ;
        R Ly =  ye_[dim_.y] - ye_[0] ;
        R Lz =  ze_[dim_.z] - ze_[0] ;

        return rti::vec3<R>(Lx, Ly, Lz);
    }

    /// Returns number of bins box
    CUDA_HOST_DEVICE
    rti::vec3<uint16_t>
    get_nxyz()
    {
        return dim_;
    }

    /// Returns position of the first corner cell position
    CUDA_HOST_DEVICE
    rti::vec3<R>
    get_origin()
    {
        return rti::vec3<R>(x_[0], y_[0], z_[0]);
    }

    /// Returns position of the first corner 
    CUDA_HOST_DEVICE
    rti::vec3<R>
    get_corner()
    {
        return rti::vec3<R>(xe_[0], ye_[0], ze_[0]);
    }

	/// Returns 8 points of bounding box
	
    /// Prints out x,y,z coordinate positions
    CUDA_HOST
    virtual void
    dump_pts()
    {
        std::cout<<"X: " ;
        for(uint16_t i=0 ; i < dim_.x ; ++i ){
            std::cout<<" " << x_[i] << " " ;
        }
        std::cout<< std::endl;

        std::cout<<"Y: " ;
        for(uint16_t i=0 ; i < dim_.y ; ++i ){
            std::cout<<" " << y_[i] << " " ;
        }
        std::cout<< std::endl;

        std::cout<<"Z: " ;
        for(uint16_t i=0 ; i < dim_.z ; ++i ){
            std::cout<<" " << z_[i] << " " ;
        }
        std::cout<< std::endl;

    }
    /// Prints out x,y,z coordinate positions
    CUDA_HOST
    virtual void
    dump_edges()
    {
        std::cout<<"X edges: " ;
        for(uint16_t i=0 ; i <= dim_.x ; ++i ){
            std::cout<<" " << xe_[i] << " " ;
        }
        std::cout<< std::endl;

        std::cout<<"Y edges: " ;
        for(uint16_t i=0 ; i <= dim_.y ; ++i ){
            std::cout<<" " << ye_[i] << " " ;
        }
        std::cout<< std::endl;

        std::cout<<"Z edges: " ;
        for(uint16_t i=0 ; i <= dim_.z ; ++i ){
            std::cout<<" " << ze_[i] << " " ;
        }
        std::cout<< std::endl;

    }


    /// Converts index of x,y,z to index of valarray(data)
    CUDA_HOST_DEVICE
    virtual inline
    uint32_t
    ijk2cnb(
        uint16_t i,
        uint16_t j,
        uint16_t k)
    {
        return k*dim_.x*dim_.y + j*dim_.x + i;
    }


    /// Writes data into file
    CUDA_HOST
    virtual void
    write_data(const std::string filename){
        std::ofstream file1( filename, std::ios::out | std::ofstream::binary);
        file1.write(reinterpret_cast<const char *>(&data_[0]), dim_.x * dim_.y * dim_.z * sizeof(T));
        file1.close();
    }

    /// Writes data in any type of valarray to a file
	/*
    CUDA_HOST
    template<class S>
    void
    write_data(std::valarray<S>& output, const std::string filename){
        std::ofstream file1( filename, std::ios::out | std::ofstream::binary);
        file1.write(reinterpret_cast<const char *>(& output[0]), output.size() * sizeof(S));
        file1.close();
    }
	*/
    /*
    /// this is not yet implemented
    #include <vtkSmartPointer.h>
    #include <vtkMetaImageWriter.h>
    #include <vtkImageReader2.h>

    CUDA_HOST
    virtual void
    write_mha(const std::string filename){

        vtkSmartPointer<vtkMetaImageWriter> writer = vtkSmartPointer<vtkMetaImageWriter>::New();
        writer->SetInputConnection(reader->GetOutputPort());
        writer->SetInpt
        writer->SetCompression(false);
        writer->SetFileName( cl_opts["--output1"][0].c_str() );
        writer->Write();

        std::ofstream file1( filename, std::ios::out | std::ofstream::binary);
        file1.write(reinterpret_cast<const char *>(&data_[0]), data_.size() * sizeof(T));
        file1.close();

    }
    */


    /// Initializes data, currently values are sum of index square for testing
    CUDA_HOST
    virtual
	void
    load_data()
    {
      this->delete_data_if_used();
      //data_ = new T[dim_.x * dim_.y * dim_.z];
    }


    /// Reads data from other source,  currently values are sum of index square for testing
    /// //total == dim_.x*dim_.y*dim_.z : not sure total is useful.
    /// //will copy
    /// in case src is CPU and dest GPU
    CUDA_HOST_DEVICE
	virtual
    void
    read_data
	(T* src)
    {
		this->delete_data_if_used();
		data_  = src;	//can be change pointer but we will copy. 
		
		int32_t total = dim_.x*dim_.y*dim_.z;		
		//printf("kernel 1: %d,  %d\n", total, src[total-1]);
		//Following two lines seem not to work in expected way
		//data_ = new T[total]{}; 
		//for(uint32_t i = 0 ; i < total ; ++i) data_[i] = src[i];
		//printf("kernel 2: %d,  %d\n", total, data_[total-1]);
    }


    /// Fills data with a given value
    CUDA_HOST_DEVICE
    virtual
	void
    fill_data(T a)
    {
		this->delete_data_if_used();
		
		//this initialization works only for first element
		//or {[ 1 ... 9] = a} seems not to work here
		data_ = new T[dim_.x * dim_.y * dim_.z]{a}; 
		for(uint32_t i = 0 ; i < dim_.x*dim_.y*dim_.z ; ++i) data_[i] = a;
    }

    /// Checks data is flipped from x_, y_, z_ and flip the order of  coordinate
    /// This method should be called after constructor gets called
    CUDA_HOST
    void
    flip_xyz_if_any(void)
    {
        flip_[0] = (x_[1] < x_[0]) ? true : false;
        flip_[1] = (y_[1] < y_[0]) ? true : false;
        flip_[2] = (z_[1] < z_[0]) ? true : false;
        if( flip_[0]){
			std::reverse( x_ , x_  + dim_.x );
			std::reverse( xe_, xe_ + dim_.x + 1 );
		} 
        if( flip_[1]){
			std::reverse( y_ , y_  + dim_.y );
			std::reverse( ye_, ye_ + dim_.y +1 );
		} 
        if( flip_[2]){
			std::reverse( z_, z_ + dim_.z );
			std::reverse( ze_, ze_ + dim_.z+1 );
		} 
    }


    /// Flip data with a given value
    CUDA_HOST
    void
    flip_data(void)
    {
        if(flip_[0] ==false &&
		   flip_[1] ==false &&
		   flip_[2] ==false)  return;
        
        std::valarray<T> tmp0(dim_.x*dim_.y*dim_.z); //temporal copy object
        tmp0 = data_;
        long int id_from = 0;
        long int id_to   = 0;

        long int idx = 0; long int idy = 0; long int idz = 0;

        for(int k =0 ; k < dim_.z ; ++k){
            for(int j=0 ; j < dim_.y ; ++j){
                for(int i=0 ; i < dim_.x ; ++i){
                    idx = (flip_[0]) ? (dim_.x-1-i)               : i;
                    idy = (flip_[1]) ? (dim_.y-1-j)*dim_.x        : j*dim_.x;
                    idz = (flip_[2]) ? (dim_.z-1-k)*dim_.x*dim_.y : k*dim_.x*dim_.y;

                    id_to    = this->ijk2cnb(i,j,k);
                    id_from  = idz + idy + idx ;
                    data_[id_to] = tmp0[id_from];
                }//x
            }//y
        }//z
    }

    /// A friend function to copy grid information of src to dest
    //template<typename T0, typename R0, typename T1, typename R1>
    //friend void clone_structure(rect3d<T0,R0>& src, rect3d<T1,R1>& dest);

    /// A friend function to interpolate new rect3d from a source
    /// interpolate(ct, dose) : possible
    /// interpolate(dose, dose)
    /// interpolate(dvf, dose) : is not possible
    //template<typename T0, typename R0, typename T1, typename R1>
    //friend void interpolate(rect3d<T0,R0>& src, rect3d<T1,R1>& dest, T1& fill_value);


    /// A friend function to warp source data (rect3d) to destination (rect3d) using dvf.
    /// It will pull src data from ref thus DVF of ref->src is neccessary.
    /// In MIM, when we calculate DIR, ref should be chosen first.
    /// It is recommended that src, dest, dvf have all same resolution.
    //template<typename T0, typename R0, typename S0>
    //friend void warp_linear( rect3d<T0,R0>& src, rect3d<T0,R0>& ref, rect3d<rti::vec3<S0>, R0>& dvf, T0 fill_value);

};

/*
// R0 and R1 should be comparable
template<typename T0, typename R0, typename T1, typename R1>
void
clone_structure
( rect3dmc<T0,R0>& src,
  rect3dmc<T1,R1>& dest)
{
    dest.dim_ = src.dim_;

    dest.x_ = new R1[dest.dim_.x];
    dest.y_ = new R1[dest.dim_.y];
    dest.z_ = new R1[dest.dim_.z];

    for(size_t i=0 ; i < dest.dim_.x ; ++i) dest.x_[i] = src.x_[i];
    for(size_t i=0 ; i < dest.dim_.y ; ++i) dest.y_[i] = src.y_[i];
    for(size_t i=0 ; i < dest.dim_.z ; ++i) dest.z_[i] = src.z_[i];
}
*/

/// Interpolates src and fill dest
/// R0 and R1 should be comparable
/*
template<typename T0, typename R0, typename T1, typename R1>
void
interpolate
( rect3d<T0,R0>& src,
  rect3d<T1,R1>& dest,
  T1& fill_value)
{

    std::cout<< src.x_[0] << ", " << src.y_[0] << ", " << src.z_[0] << std::endl;
    std::cout<< dest.x_[0] << ", " << dest.y_[0] << ", " << dest.z_[0] << std::endl;

    ///< Number of voxels: Nx, Ny, Nz
    const size_t nX = dest.dim_.x;
    const size_t nY = dest.dim_.y;
    const size_t nZ = dest.dim_.z;

    ///< center point of destination
    dest.data_.resize(nX*nY*nZ);

    rti::vec3<R1> p = { dest.x_[0], dest.y_[0], dest.z_[0] };
    size_t counter = 0;
    for(size_t k = 0 ; k < nZ ; ++k){
        p.z = dest.z_[k] ;
        for(size_t j=0 ; j < nY ; ++j){
            p.y = dest.y_[j] ;
            for(size_t i=0; i < nX ; ++i){
                p.x = dest.x_[i] ;
                dest.data_[dest.ijk2cnb(i,j,k)] =
                src.is_in_point(p) ? src(p) : fill_value;
            }//x
        }//y
    }//z

}
*/

/// Warping data in src using vector field
/*
template<typename T0, typename R0, typename S0>
void
warp_linear
(rect3d<T0,R0>& src,
 rect3d<T0,R0>& dest,
 rect3d<rti::vec3<S0>, R0>& vf,
 T0 fill_value)
{
    ///< Number of voxels: Nx, Ny, Nz
    const size_t nX = dest.dim_.x;
    const size_t nY = dest.dim_.y;
    const size_t nZ = dest.dim_.z;

    dest.data_.resize(nX*nY*nZ);

    ///< Looping reference
    rti::vec3<R0> p_dest(dest.x_[0], dest.y_[0], dest.z_[0]);

    for(size_t k = 0 ; k < nZ ; ++k){
        p_dest.z = dest.z_[k];
        for(size_t j = 0 ; j < nY ; ++j){
            p_dest.y = dest.y_[j];
            for(size_t i = 0 ; i < nX ; ++i){
                p_dest.x = dest.x_[i];

                T0 value ;
                if (vf.is_in_point(p_dest)){
                    /// If destination point is in DVF grid points,
                    /// apply translation and then check new position is in source
                    /// Then, assign value by interpolating values at 8 coners surrounding new position.
                    rti::vec3<R0> p_new = p_dest + vf(p_dest);
                    value = src.is_in_point(p_new) ? src(p_new) : fill_value ;
                }else{
                    value = src.is_in_point(p_dest) ? src(p_dest) : fill_value;

                }//vf.is_in_point
                dest.data_[dest.ijk2cnb(i,j,k)] = value;

            }//x
        }//y
    }//z

}
*/

}

#endif
