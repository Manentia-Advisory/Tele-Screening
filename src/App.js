import { Route, Switch } from 'react-router-dom';

import ImageUploadPage from "./pages/image-upload-page.component";
import SignUpPage from "./pages/sign-in-page.component";


function App() {
  return (
    <div>
      <Switch>
        <Route exact path='/' component={SignUpPage}/>
        <Route path="/app" component={ImageUploadPage}/>
      </Switch>
    </div>
  );
}

export default App;
