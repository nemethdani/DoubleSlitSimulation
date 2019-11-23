//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Nemeth Daniel
// Neptun : FTYYJR
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"
//#include <iostream>

//================
// Okos Float osztály CPP11 labor megoldásaiból: https://cpp11.eet.bme.hu/lab03/#4
// Nagy részét magam is mefírtam, itt az összehasonlításokhoz fogom használni
namespace smartfloat{
	class Float {
	public:
		Float() = default;
		Float(float value) : value_(value) {}
		explicit operator float() const { return value_; }
	
		static constexpr float epsilon = 1e-7f;
	
	private:
		float value_;
	};
	
	
	Float operator+(Float f1, Float f2) {
		return float(f1) + float(f2);
	}
	
	Float & operator+=(Float &f1, Float f2) {
		return f1 = f1 + f2;
	}
	
	Float operator-(Float f1, Float f2) {
		return float(f1) - float(f2);
	}
	
	Float & operator-=(Float &f1, Float f2) {
		return f1 = f1 - f2;
	}
	Float operator/(Float f1, Float f2) {
		return float(f1) / float(f2);
	}
	
	Float & operator/=(Float &f1, Float f2) {
		return f1 = f1 / f2;
	}
	Float operator*(Float f1, Float f2) {
		return float(f1) * float(f2);
	}
	
	Float & operator*=(Float &f1, Float f2) {
		return f1 = f1 * f2;
	}
	
	/* egyoperandusú */
	Float operator-(Float f) {
		return -float(f);
	}
	
	/* kisebb */
	bool operator<(Float f1, Float f2) {
		return float(f1) < float(f2)-Float::epsilon;
	}
	
	/* nagyobb: "kisebb" fordítva */
	bool operator>(Float f1, Float f2) {
		return f2<f1;
	}
	
	/* nagyobb vagy egyenlő: nem kisebb */
	bool operator>=(Float f1, Float f2) {
		return !(f1<f2);
	}
	
	/* kisebb vagy egyenlő: nem nagyobb */
	bool operator<=(Float f1, Float f2) {
		return !(f1>f2);
	}
	
	/* nem egyenlő: kisebb vagy nagyobb */
	bool operator!=(Float f1, Float f2) {
		return f1 > f2 || f1 < f2;
	}
	
	/* egyenlő: nem nem egyenlő */
	bool operator==(Float f1, Float f2) {
		return !(f1 != f2);
	}
	
	

}
// =====================
using namespace smartfloat;

bool operator==(const vec2& v1, const vec2& v2){
	return Float(v1.x)==Float(v2.x) && (Float(v1.y)==Float(v2.y));
}
bool operator==(const vec3& v1, const vec3& v2){
	return Float(v1.x)==Float(v2.x) && (Float(v1.y)==Float(v2.y)) && Float(v1.z)==Float(v2.z);
}


// Az alábbiakat a raytracing mintaprogram módosítával készítettem

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	float reffractiveIndex;
	bool rough, reflective, refractive;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { 
		shininess = _shininess;
		if(_kd==vec3(0,0,0) && _ks==vec3(0,0,0)){
			rough=false;
		}
		else rough=true;
		reflective=refractive=false;
		reffractiveIndex=1.000293; //air
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Transformable{
	protected:
		mat4 transformMatrix;
	public:
		Transformable(const mat4& transformMatrix=TranslateMatrix(vec3(0,0,0))):transformMatrix(transformMatrix){};
};

class Intersectable: public Transformable {
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
	Intersectable(const mat4& transformMatrix):Transformable(transformMatrix){};
};

// struct Sphere : public Intersectable {
// 	vec3 center;
// 	float radius;

// 	Sphere(const vec3& _center, float _radius, Material* _material) {
// 		center = _center;
// 		radius = _radius;
// 		material = _material;
// 	}

// 	Hit intersect(const Ray& ray) {
// 		Hit hit;
// 		vec3 dist = ray.start - center;
// 		float a = dot(ray.dir, ray.dir);
// 		float b = dot(dist, ray.dir) * 2.0f;
// 		float c = dot(dist, dist) - radius * radius;
// 		float discr = b * b - 4.0f * a * c;
// 		if (discr < 0) return hit;
// 		float sqrt_discr = sqrtf(discr);
// 		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
// 		float t2 = (-b - sqrt_discr) / 2.0f / a;
// 		if (t1 <= 0) return hit;
// 		hit.t = (t2 > 0) ? t2 : t1;
// 		hit.position = ray.start + ray.dir * hit.t;
// 		hit.normal = (hit.position - center) * (1.0f / radius);
// 		hit.material = material;
// 		return hit;
// 	}
// };

struct mat2{
	mat2(){};
	float matrix[2][2];
	mat2(float m00, float m01, float m10, float m11){
		matrix[0][0]=m00; matrix[0][1]=m01;
		matrix[1][0]=m10; matrix[1][1]=m11;
	}
	mat2 operator*(const mat2& right) const {
		mat2 result;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				result.matrix[i][j] = 0;
				for (int k = 0; k < 2; k++) result.matrix[i][j] += matrix[i][k] * right.matrix[k][j];
			}
		}
		return result;
	}
	mat2 operator+(const mat2& right) const {
		mat2 result;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				result.matrix[i][j] = matrix[i][j]+right.matrix[i][j];
				
			}
		}
		return result;
	}
	mat2 operator*(float a)const{
		mat2 r;
		for(int i=0;i<2;++i){
			for(int j=0;j<2;++j){
				r.matrix[i][j]=matrix[i][j]*a;
			}
		}
		return r;
}
	mat2 operator-(const mat2& right)const{
		mat2 r=right*float(-1);
		return *this+r;
	}
	mat2 operator-()const{
		return (*this)*float(-1);
	}

};


mat2 invert(const mat2& mat){
	float a=mat.matrix[0][0];
	float b=mat.matrix[0][1];
	float c=mat.matrix[1][0];
	float d=mat.matrix[1][1];
	const mat2 temp{d, -b, -c, a};
	float mult=1/(a*d-b*c);
	mat2 t=temp*mult;
	return t;
}

mat4 Mat4(mat2 A, mat2 B, mat2 C, mat2 D){
	mat4 r;
	r.m[0][0] = A.matrix[0][0]; r.m[0][1] = A.matrix[0][1]; r.m[0][2] = B.matrix[0][0]; r.m[0][3] =B.matrix[0][1];
	r.m[1][0] = A.matrix[1][0]; r.m[1][1] = A.matrix[1][1]; r.m[1][2] = B.matrix[1][0]; r.m[1][3] = B.matrix[1][1];
	r.m[2][0] = C.matrix[0][0]; r.m[2][1] = C.matrix[0][1]; r.m[2][2] = D.matrix[0][0]; r.m[2][3] = D.matrix[0][1];
	r.m[3][0] = C.matrix[1][0]; r.m[3][1] = C.matrix[1][1]; r.m[3][2] =D.matrix[1][0]; r.m[3][3] = D.matrix[1][1];
	return r;
}

bool operator==(const mat4& a, const mat4& b){
	for(int i=0;i<4;++i){
		for(int j=0;j<4;++j){
			if(Float(a.m[i][j])!=Float(b.m[i][j])) return false;
		}
		
	}
	return true;
}

// https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion
mat4 invert(mat4 mat){
	mat2 A=mat2{mat.m[0][0], mat.m[0][1],
				mat.m[1][0], mat.m[1][1]};
	mat2 B=mat2{mat.m[0][2], mat.m[0][3],
				mat.m[1][2], mat.m[1][3]};
	mat2 C=mat2{mat.m[2][0], mat.m[2][1],
				mat.m[3][0], mat.m[3][1]};
	mat2 D=mat2{mat.m[2][2], mat.m[2][3],
				mat.m[3][2], mat.m[3][3]};
	mat2 A_1=invert(A);

	mat2 A_=A_1+A_1*B*invert(D-C*A_1*B)*C*A_1;
	mat2 B_=-A_1*B*invert(D-C*A_1*B);
	mat2 C_=-invert(D-C*A_1*B)*C*A_1;
	mat2 D_=invert(D-C*A_1*B);

	mat4 r=Mat4(A_, B_, C_, D_);
	return r;

	
}

mat4 transpose(const mat4& mat){
	mat4 r;
	for(int i=0;i<4;++i){
		for (int j=0;j<4;++j){
			r.m[i][j]=mat.m[j][i];
		}
	}
	return r;
}

mat4 operator*(float f, const mat4& m){
	mat4 ret;
	for(int i=0;i<4;++i){
		for(int j=0;j<4;++j){
			ret.m[i][j]=f*m.m[i][j];
		}
	}
	return ret;
}



class QuadricSurface: public Intersectable{
	mat4 matrix;
	

	// https://marctenbosch.com/photon/mbosch_intersection.pdf
	vec3 normal(vec3 position)const{
		float n[3];
		vec4 pos(position.x, position.y, position.z, 1);
		for(int i=0;i<3;++i){
			n[i]=dot(pos, vec4(matrix.m[i][0], matrix.m[i][1],matrix.m[i][2],matrix.m[i][3]));
		}
		vec3 N=2*vec3(n[0], n[1], n[2]);
		return normalize(N);
	}

	public:
		QuadricSurface(mat4 matrix, mat4 transformMatrix, Material* _material):
			Intersectable(transformMatrix),
			matrix(matrix)
				{material=_material;};

		Hit intersect(const Ray& ray) override{
			mat4 T_1=invert(transformMatrix);
			mat4 Q=T_1*matrix*transpose(T_1);
			vec4 S(ray.start.x, ray.start.y, ray.start.z, 1);

			float A=0, B1=0, B2=0, C=0;
			{
				float a[3];
				for(int j=0;j<3;++j){
					vec3 c(Q.m[0][j], Q.m[1][j] , Q.m[2][j]);
					a[j]=dot(ray.dir, c);
				}
				A=dot(vec3(a[0], a[1], a[2]), ray.dir);

				
				float b1[3];
				for(int j=0;j<3;++j){
					vec4 c(Q.m[0][j], Q.m[1][j] , Q.m[2][j], Q.m[3][j]);
					b1[j]=dot(S, c);
				}
				B1=dot(vec3(b1[0], b1[1], b1[2]), ray.dir);

				
				float b2[4];
				for(int j=0;j<4;++j){
					vec3 c(Q.m[0][j], Q.m[1][j] , Q.m[2][j]);
					b2[j]=dot(ray.dir, c);
				}
				B2=dot(vec4(b2[0], b2[1], b2[2], b2[3]), S);

				
				float c[4];
				for(int j=0;j<4;++j){
					vec4 cc(Q.m[0][j], Q.m[1][j] , Q.m[2][j], Q.m[3][j]);
					c[j]=dot(S, cc);
				}
				C=dot(vec4(c[0], c[1], c[2], c[3]), S);
			}

			Hit hit;
			float B=B1+B2;
			float discr=B*B-4.0*A*C;
			if (discr < 0) return hit;
			float sqrt_discr = sqrtf(discr);
			Float t1 = (-B + sqrt_discr) / 2.0f / A;	// t1 >= t2 for sure
			if (t1 <= 0) return hit;
			Float t2 = (-B - sqrt_discr) / 2.0f / A;
			hit.t = (t2 > Float(0)) ? float(t2) : float(t1);
			hit.position = ray.start + ray.dir * hit.t;

			hit.normal = normal(hit.position);

			hit.material = material;
			return hit;
		};
};

class Cilinder: public QuadricSurface{
	public:
		Cilinder(mat4 transformMatrix, Material* _material):
			QuadricSurface((-1.0f)*ScaleMatrix(vec3(-1, -1, 0)), transformMatrix, _material){};
};

class Hyperboloid_ofOneSheet: public QuadricSurface{
	public:
		Hyperboloid_ofOneSheet(mat4 transformMatrix, Material* _material):
			QuadricSurface( (-1.0f)*ScaleMatrix(vec3(-1, -1, 1)), transformMatrix, _material){};
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light: public Transformable{
	virtual vec3 incidentRadiance(Hit hit)const=0;
	virtual vec3 direction(Hit hit)const=0;
	virtual vec3 center()const=0;
	vec3 Le;
	using Transformable::Transformable;
};

struct DirectionalLight: public Light {
	vec3 dir;
	DirectionalLight(vec3 _direction, vec3 _Le, const mat4& transformMatrix=TranslateMatrix(vec3(0,0,0))): Light(transformMatrix) {
		dir= normalize(_direction);
		Le = _Le;
	}
	vec3 incidentRadiance(Hit hit)const override{return Le;}
	vec3 direction(Hit hit)const override{return dir;}
	vec3 center()const override{
		float inf=1.0f/0.0f;
		return vec3(inf, inf, inf);
	}
};

class Curve{
	protected:
		std::vector<vec2> wCtrlPoints;
	public:
		Curve(std::vector<vec2> ctrl):wCtrlPoints(ctrl){};
};

//tantárgyi segédanyag
//Bezier using Bernstein polynomials
class BezierCurve : public Curve {
	float B(int i, float t) {
		int n = wCtrlPoints.size() - 1; // n deg polynomial = n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}
public:
	using Curve::Curve;
	vec2 r(float t) {
		vec2 wPoint = vec2(0, 0);
		for (unsigned int n = 0; n < wCtrlPoints.size(); n++) wPoint = wPoint+ wCtrlPoints[n] * B(n, t);
		return wPoint;
	}
};

// http://cg.iit.bme.hu/portal/sites/default/files/oktatott%20t%C3%A1rgyak/sz%C3%A1m%C3%ADt%C3%B3g%C3%A9pes%20grafika/geometri%C3%A1k%20%C3%A9s%20algebr%C3%A1k/complex.cpp
//--------------------------
struct Complex {
//--------------------------
	float x, y;

	Complex(float x0 = 0, float y0 = 0) { x = x0, y = y0; }
	Complex operator+(Complex r) { return Complex(x + r.x, y + r.y); }
	Complex operator-(Complex r) { return Complex(x - r.x, y - r.y); }
	Complex operator*(Complex r) { return Complex(x * r.x - y * r.y, x * r.y + y * r.x); }
	Complex operator/(Complex r) {
		float l = r.x * r.x + r.y * r.y;
		return (*this) * Complex(r.x / l, -r.y / l);
	}
};

Complex Polar(float r, float phi) {
	return Complex(r * cosf(phi), r * sinf(phi));
}

vec4 centroid(const std::vector<vec4>& v){
			vec4 centroid(0,0,0,0);
			for(auto l: v) centroid=centroid+l;
			centroid=centroid/v.size();
			return centroid;
}
const float epsilon = 0.0001f;
class TwoSlitLight: public Light{
	std::vector<vec4> lampsXYZ;
	float Amplitude=0.01;
	vec4 cntr; //pontszerű fényforrás
	float wavelength=0.526; //um=micrometer, mindenhol ezt a mértékegységet használom, hondolom itt is ez kell
	vec3 rgb=vec3(78.0f, 1.0f, 0.0f); //rgb(78,255, 0): kell normalizálni? számítás: https://academo.org/demos/wavelength-to-colour-relationship/
	public:
		TwoSlitLight(float targetRadiusOfCilinder=6, float targetUdistance=3, const mat4& transformMatrix=TranslateMatrix(vec3(0,0,0))):
			Light(transformMatrix)
			{
			float baseUdistance=targetUdistance/targetRadiusOfCilinder;
			float phi=2*asinf(baseUdistance/2);
			std::vector<vec2> ctrpoints={vec2(-phi/2, 0.05f), vec2(-phi/2, -0.05f), vec2(phi/2, -0.05f), vec2(phi/2, 0.05f)};
			BezierCurve bc(std::move(ctrpoints));
			std::vector<vec2> lamps; // angle, Z
			lamps.reserve(100);
			for(Float t=0.0f;t<=99.0f/100.0f; t+=1.0f/(100.0f-1.0f))
				lamps.push_back(bc.r(float(t)));
			
			lampsXYZ.reserve(100);
			for(vec2 l:lamps){
				lampsXYZ.emplace_back(cosf(l.x), sinf(l.x), l.y, 1);
			}

			cntr=centroid(lampsXYZ);
			cntr=cntr*TranslateMatrix(vec3(1-cosf(phi/2)+epsilon,0,0)); //eltoljuk, hogy a henger szélén legyen(ne benne)
			cntr=cntr*transformMatrix;
			for(auto&& l:lampsXYZ)
				l=l*transformMatrix;
			
			

		}
		vec3 incidentRadiance(Hit hit)const override{
			Complex superposition;
			for(vec4 lamp: lampsXYZ){
				float distance=length(hit.position-vec3(lamp.x, lamp.y, lamp.z));
				float k=2*(M_PI)/wavelength; 
				superposition=superposition+Polar(Amplitude/distance, k*distance); //t=0 => omega*t=0, fáziseltolódás=0 
			}
			
			float energy=superposition.x*superposition.x+superposition.y*superposition.y; //R^2=x^2+y^2
			vec3 rad=energy*rgb;
			return rad;
		}
		vec3 direction(Hit hit)const override{
			vec3 ret= vec3(cntr.x, cntr.y, cntr.z)-hit.position;
			return ret;
		}
		vec3 center()const override{
			return vec3(cntr.x, cntr.y, cntr.z);
		}
};

float rnd() { return (float)rand() / RAND_MAX; }





class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;

	bool shadowIntersect(Ray ray)const {	// for directional lights
		for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 DirectLight(const Hit& hit, const Ray& ray)const{
		vec3 outRadiance=hit.material->ka * La;
		for(Light* light:lights){
			float cosTheta = dot(hit.normal, light->direction(hit));
			vec3 r=hit.position + hit.normal * epsilon;
			Ray shadowRay(r, light->direction(hit));
			Hit shadowHit=firstIntersect(shadowRay);
			if (cosTheta > 0 && shadowHit.t<0 || shadowHit.t>length(r-light->center())) {	// shadow computation
	 			outRadiance = outRadiance + light->incidentRadiance(hit) * hit.material->kd * cosTheta;
	 			vec3 halfway = normalize(-ray.dir + light->direction(hit));
	 			float cosDelta = dot(hit.normal, halfway);
	 			if (cosDelta > 0){
					
					 float pow=powf(cosDelta, hit.material->shininess);
					 vec3 ir_ks=light->incidentRadiance(hit) * hit.material->ks;
					 outRadiance = outRadiance + ir_ks * pow;
				 }
	 		}
			
		}
		return outRadiance;
	};

	vec3 reflect(vec3 inDir, vec3 normal) {
		return inDir - normal * dot(normal, inDir) * 2.0f;
	};

	// vec3 Fresnel(vec3 inDir, vec3 normal, Hit hit) {
	// 	float cosa = -dot(inDir, normal); 
	// 	vec3 one(1, 1, 1);
	// 	float airReffractiveIndex=1.000293; //air
	// 	float materialRefreactiveIndex=(*(hit.material)).reffractiveIndex;
	// 	float n_=materialRefreactiveIndex/airReffractiveIndex;
	// 	vec3 n(n_, n_, n_);
	// 	vec3 F0 = ((n - one)*(n - one) + kappa*kappa) /	((n+one)*(n+one) + kappa*kappa); 
	// 	return F0 + (one – F0) * pow(1-cosa, 5); 
	// }

public:
	void build() {
		vec3 eye = vec3(6, 0, 0), vup = vec3(0, 0, 1), lookat = vec3(30, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new DirectionalLight(lightDirection, Le));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(0.0f, 0.0f, 0.0f);
		Material * material = new Material(kd, ks, 50);
		// for (int i = 0; i < 500; i++) 
		// 	objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, material));

		mat4 cylinderTransform=ScaleMatrix(vec3(6,6,1));
		objects.push_back(new Cilinder{cylinderTransform, material});
		
		lights.push_back(new TwoSlitLight{6, 3, cylinderTransform });
		objects.push_back(new Hyperboloid_ofOneSheet{ScaleMatrix(vec3(6,6,30))*TranslateMatrix(vec3(30, 0,0)), material});
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
				//printf("%d" "%d" "%s", X, X, "rendered\n");
			}
		}
	}

	Hit firstIntersect(Ray ray)const {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	

	// vec3 trace(Ray ray, int depth = 0) {
	// 	Hit hit = firstIntersect(ray);
	// 	if (hit.t < 0) return La;
	// 	vec3 outRadiance = hit.material->ka * La;
	// 	for (Light * light : lights) {
	// 		Ray shadowRay(hit.position + hit.normal * epsilon, light->direction(hit));
	// 		float cosTheta = dot(hit.normal, light->direction(hit));
	// 		if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
	// 			outRadiance = outRadiance + light->incidentRadiance(hit) * hit.material->kd * cosTheta;
	// 			vec3 halfway = normalize(-ray.dir + light->direction(hit));
	// 			float cosDelta = dot(hit.normal, halfway);
	// 			if (cosDelta > 0) outRadiance = outRadiance + light->incidentRadiance(hit) * hit.material->ks * powf(cosDelta, hit.material->shininess);
	// 		}
	// 	}
	// 	return outRadiance;
	// }

	vec3 trace(Ray ray, int d=0, int maxdepth=8) {
		if (d > maxdepth) return La;
		Hit hit = firstIntersect(ray);
		if(hit.t < 0) return La; // nothing
		vec3 outRad(0, 0, 0);
		if(hit.material->rough) outRad  =  DirectLight(hit, ray);

		// if(hit.material->reflective){
		// 	vec3 reflectionDir = reflect(ray.dir,hit.normal);
		// 	Ray reflectRay(hit.position + hit.normal * epsilon, reflectionDir);
		// 	outRad += trace(reflectRay,d+1)*Fresnel(ray.dir,N);
		// }
		// if(hit.material->refractive) {
		// 	ior = (ray.out) ? n.x : 1/n.x;
		// 	vec3 refractionDir = refract(ray.dir,N,ior);
		// 	if (length(refractionDir) > 0) {
		// 	Ray refractRay(r - N, refractionDir, !ray.out);
		// 	outRad += trace(refractRay,d+1)*(vec3(1,1,1)–Fresnel(ray.dir,N));
		// 	}
		// }
		return outRad;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image) 
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

void invertTest(){
	mat4 a{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
		if(!(invert(a)*a == TranslateMatrix(vec3(0,0,0)))){
			printf("%s", "hibás invertálás\n");
		}
		else printf("%s", "inverting OK\n");
}

// Initialization, create an OpenGL context
void onInitialization() {

	//invertTest();

	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
