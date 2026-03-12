# 1) 
Rethink challanges mentioned in the introduction, they should be basically main axis of the work we need to reference it in the results and in the discussion to keep the same order as we are testing and discussing current state of the art in light of them 
Becouse of tat we need to rethink a bit how we are designing those challenges as we need to have experiment for each of them  ;  we are running new experiment with sciml as you can see in repository codefor it to calculate attenuation correction  this is basically to showcase the power of scientific machine learning and that this ecosystem is quite unique to Julia so medimages give better access to it thanks to being basic medical imaging library ; We also have other things that we need to underline in this challanges and showcase 
a) volume - just to show that we have a lot of diffrent 3d modalities in nuclear medicine and it put stress on needed computation 
b) speed speed needed becouse of volume and big image with big ranges (here possible becouse of julia and kernels)
c) differentiability - this is a plus making registration frameworks as well as optimizing augmentation better
d) the access of julia ecosystem - we had started from 
e) metadata management - hard and tricky task in medical imaging as image here is not only tensor it has real physical interpretaiton

# 2) 
Analize "elsarticle/sources/1-s2.0-S016926072300041X-main.pdf"  try to replicate its style and structure ; also reference it in article as \cite{Rufenacht_2023}

# 3) 
at the of introduction currently we have """MedImages.jl addresses ... """ or """MedImages.jl implements a BIDS-inspired metadata """ or """MedImages.jl leverages this capability directly'"""   generally parts like this are better to put into introduction but still we need to have some smoother change from introduction to architecture 

# 4) 
Subsection """Clinical Applications of SUV'""" should be removed as susection put shortly this information in table or something like that 

# 5)
 analize  """https://github.com/BioMedIA/deepali""" and """https://biomedia.github.io/deepali/index.html"""   it is generally specialized research library that focuses almost exclusively on image registration and spatial transformations in PyTorch. 1  Developed by the Heartflow-Imperial College London research lab, it is designed for users who need fine-grained control over transformation models like B-splines and Free-Form Deformations (FFD) Unlike MONAI or TorchIO, which treat augmentations as a means to improve model generalization, Deepali treats the transformation itself as the primary object of interest. Its SpatialTensor and Grid abstractions allow for the representation of complex sampling grids that are fully differentiable. This is particularly useful for building "Image-and-Spatial Transformer Networks," where the goal is to learn a mapping that aligns two images while simultaneously estimating the underlying deformation field.Deepali's commitment to differentiability is absolute; almost every component—from loss functions (like Normalized Mutual Information) to coordinate space mappings—is built to support torch.autograd. However, this comes at the cost of a steeper learning curve, as the library requires a deeper understanding of coordinate space theory and parametric modeling than the more accessible TorchIO.

cite it in text by \cite{schuh_2026_18370988}

Also mention that deepali is not for typical segmentation pipelines and remain quite specialized and is available only to researchers using pytorch 

# 6) 
Analize elsarticle/sources/306951v2.full.pdf and cite it by \cite{Esteban_2018} particularly it is important to cite from it the information about lack of standarization and multiple diffrent preprocessings - so something we want to solve from the begining by creating standardized medical imaging format ; when analizing article remember we are focusing on nuclear medicine but from format perspective it is relatively similar to other modalities  just keep in mind our main purpose is CT PET spect dosemaps ... 
Describe we want to avoid fragmentations ...  metadata drifts ; explain how metadata drift is dangerous for ML models 

# 7) 
Analize also elsarticle/sources/note_geminie.txt it seem to have some intresting ideas 

# 8) 
add more details for benchmarks for example 100 case benchmark why we chosen it like that - becouse it is typical workflow in multimodality framework where diffrent modalities need to be resampled to one and have the same range standardize orientation standardize spacing among dataset ... - explain why it is important Explain also where we used CPU and where GPU ; and that speed is primarly related to highly fused and optimised gpu kernels 

# 9)
in """Due to Julia's high-precision handling of temporal and clinical metadata, MedImages.jl achieves identical results for SUVbw (Body Weight) calculations, matching SimpleITK's output up to $10^{-14}$ in double precision. This ensures that the massive speedups achieved through GPU acceleration do not compromise clinical validity.""" mention that we can proove it as we open source experiments  

# 10)
in experiments based on code in this repo underline what was done on single cases what batched what on whole dataset 

# 11)
 when we are referencing autopet dataset cite it by \cite{Dexl_2025}

# 12) 
when you cite sciml  cite \cite{Berman_2024} analize this article """elsarticle/sources/2410.10908v2.pdf'""" it also has some references to julia and 2 language problem incorporate it and cite this paper 

# 13) 
generate and embed plot for """ The original Mean SUV was 3.4913 with a volume of 52.55 $\text{cm}^3$, and the final transformed volume exhibited a Mean SUV of 3.4660 and a volume of 51.86 $\text{cm}^3$""" 

# 14) 
before """ Scientific Machine Learning (SciML) is a paradigm that embeds mechanistic, physics-based models---such as ordinary and partial differential equations, conservation laws, and domain-specific constraints... """ we need something like """Nuclear medicine imaging is inherently quantitative, but its key readouts (e.g., SUV, time–activity curves, absorbed dose) are highly sensitive to geometric preprocessing such as resampling, rotation, and registration. In practice, these operations must often be integrated into AI pipelines for motion correction, longitudinal alignment, and learned preprocessing, where transform parameters are optimized jointly with network weights. However, many commonly used imaging backends execute spatial resampling outside the automatic differentiation graph, preventing end-to-end training through geometric steps. To demonstrate that MedImages.jl enables fully differentiable volumetric spatial operations suitable for nuclear medicine workflows, we implement a spatial transformer that learns to invert unknown 3D rotations and quantify its reconstruction improvement.""" 
# 15) 
based on """elsarticle/sources/DDA_survey_TPAMI_2022.pdf"""  that you should cite by \cite{Shi_2023} when using its information add more why differentiable augmantation is rare hard and usefull 

# 16)
 try to simplify language and explain all hard concepts so both nuclear medicine physycians and ml engineers will be able to understand all ; for example """Furthermore, MONAI's metadata management is often a reactive layer; many of its core geometric transformations default to backends like SciPy or SimpleITK, meaning the actual computation occurs in a non-differentiable "black box," breaking the autograd graph.""" may be hard for nuclear medicine physycians 
or for example """TorchIO approaches the problem through a hierarchical Subject class system. While effective for multi-modal synchronization, TorchIO is primarily a manager of third-party backends (SimpleITK, NiBabel). This dependency chain maintains a layer of indirection that can lead to memory management issues in large-scale training. """ maybe more like """ A similar design pattern appears in TorchIO, which prioritizes convenient multimodal coordination but still builds its core processing around external imaging backends. TorchIO approaches the problem through a hierarchical Subject class system. While effective for multi-modal synchronization, TorchIO is primarily a manager of third-party backends (SimpleITK, NiBabel). This dependency chain maintains a layer of indirection that can lead to memory management issues in large-scale training. (citation)"""

# 17) 
using """"https://www.overleaf.com/learn/latex/Glossaries""" get all acronyms properly formatted and explained like ODE ... 

# 18) 
Make sure to explain that this is general purpose library but thanks to it it can be used to many things like preprocessing loading preparing for visualization and thanks to differentiability of operations to registration and differentiability of augmentations to training models ... 
