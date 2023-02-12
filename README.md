# Fleet EV
This project was inspired by the growing popularity of electric vehicles (EVs) and the increasing need for mass adoption of sustainable transportation. As members of the team had experience working in the EV industry and facing the challenges of charging an electric vehicle, we decided to tackle the challenge of route planning for a fleet of EVs, with the goal of minimizing.

# app.py
On the app development front, we faced challenges with data/algorithm integration and user interface design. We initially attempted to gather data from a Swedish company's API, but were unable to gain access over the weekend. As a result, we had to find alternative, free sources of data for our app, such as the OpenChargeMap API and ElectricityMaps API. We also had to design a user-friendly interface that could effectively display charging information for multiple EVs via the use of Streamlit.

# grid_solve.py 
In developing the algorithm, we solve a vehicle routing with battery constraint problem, which unfortunately is NP-Hard in general. To achieve a solution efficiently and near-optimally, we use a mix of mixed integer linear programming and heuristics methods, which allow us to minimize our fleet’s CO2 consumption, energy expenditure, and travel time. Through this process, we learned about the nuances of developing efficient and effective optimization models, as well as the importance of balancing different objectives in a practical and realistic way.

# final_solver.ipynb
Used to call the libraries.
