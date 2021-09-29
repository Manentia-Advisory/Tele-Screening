import { Route, Switch } from 'react-router-dom';

import ImageUploadPage from "./pages/image-upload-page.component";
import ResultPage from './pages/result-page.component';
import SignUpPage from "./pages/sign-in-page.component";


function App() {
  return (
    <div>
      <Switch>
        <Route exact path='/login' component={SignUpPage}/>
        <Route path="/" component={ImageUploadPage}/>
        <Route path="/result" component={ResultPage}/>
      </Switch>
    </div>
  );
}

export default App;
